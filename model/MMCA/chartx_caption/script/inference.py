import os
import sys
import json
import torch
from PIL import Image
from tqdm import tqdm

# ========= 环境设置 =========
sys.path.append("/data/jguo376/project/model/mPLUG-Owl/mPLUG-Owl")
torch.set_grad_enabled(False)

from transformers import AutoTokenizer
from mplug_owl.modeling_mplug_owl import MplugOwlForConditionalGeneration
from mplug_owl.processing_mplug_owl import MplugOwlProcessor, MplugOwlImageProcessor
from peft import PeftModel

# ========= 路径配置 =========
model_path_for_weights = "/data/jguo376/pretrained_models/mmca_merged_model"
processor_ref_path = "/data/jguo376/pretrained_models/mplug-owl-llama-7b"
lora_path = "/data/jguo376/project/model/MMCA/fine-tuning/output/sft_v0.1_ft_chartentail/checkpoint-6100"

input_json = "/data/jguo376/project/dataset/test_dataset/ChartX/test_eva_data/data/eva_test.json"
output_json = "/data/jguo376/project/model/MMCA/chartx_entail_singal/chartx.json"
chart_root = "/data/jguo376/project/dataset/ChartX_dataset/ChartX"

# ========= 控制推理范围 =========
start_index = 0        # ✅ 从第几条开始处理
max_items = None      # ✅ 最多处理多少条（None 表示全部）

# ========= 加载模型 =========
print("📦 加载基础模型权重...")
model = MplugOwlForConditionalGeneration.from_pretrained(
    model_path_for_weights,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

if lora_path:
    print(f"🪄 加载额外 LoRA adapter: {lora_path}")
    model = PeftModel.from_pretrained(model, lora_path)

# ========= 加载 processor =========
tokenizer = AutoTokenizer.from_pretrained(processor_ref_path)
image_processor = MplugOwlImageProcessor.from_pretrained(processor_ref_path)
processor = MplugOwlProcessor(image_processor, tokenizer)

# ========= 推理参数 & prompt =========
generate_kwargs = {
    'do_sample': True,
    'top_k': 5,
    'max_length': 512
}
query_prompt = """The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
Human: <image>
Human: Please describe the chart.
AI:"""

# ========= 加载输入数据 =========
with open(input_json, "r", encoding="utf-8") as f:
    data_list = json.load(f)

# ========= 断点续跑支持 =========
processed_imgs = set()
if os.path.exists(output_json):
    with open(output_json, "r", encoding="utf-8") as f:
        try:
            existing_results = json.load(f)
            for entry in existing_results:
                processed_imgs.add(entry["img"])
        except Exception:
            existing_results = []
else:
    existing_results = []

results = existing_results.copy()

# ========= 批量推理主循环 =========
print("🚀 开始批量生成图表描述...")
selected_data = data_list[start_index:]
if max_items is not None:
    selected_data = selected_data[:max_items]

count = 0
for item in tqdm(selected_data, desc="Generating captions"):
    rel_path = item.get("img")
    image_path = os.path.join(chart_root, rel_path.lstrip("./"))

    if image_path in processed_imgs:
        continue

    if not os.path.exists(image_path):
        caption = f"[ERROR] Image not found: {image_path}"
    else:
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = processor(text=[query_prompt], images=[image], return_tensors="pt")
            inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                output = model.generate(**inputs, **generate_kwargs)
                caption = tokenizer.decode(output[0], skip_special_tokens=True)
        except Exception as e:
            caption = f"[ERROR] {str(e)}"

    item["model_name"] = "mPLUG-Owl + MMCA + LoRA"
    item["img"] = image_path
    item["generated_caption"] = caption
    results.append(item)
    processed_imgs.add(image_path)
    count += 1

    print(f"[✓] {item.get('imgname')}")
    print(f"    → {caption}\n")

    # 中间保存
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

# ========= 最终保存 =========
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n✅ 推理完成！共生成 {len(results)} 条图表描述，结果保存至：{output_json}")
