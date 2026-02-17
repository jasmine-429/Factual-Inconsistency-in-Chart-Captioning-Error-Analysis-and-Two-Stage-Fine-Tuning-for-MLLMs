import os
import sys
import json
import torch
from PIL import Image
from tqdm import tqdm
import traceback

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

input_jsons = [
    "/data/jguo376/project/dataset/chartsumm/test_k.json",
    "/data/jguo376/project/dataset/chartsumm/test_s.json"
]
output_jsons = [
    "/data/jguo376/project/model/MMCA/chartsumm_entail_singal/test_k_output.json",
    "/data/jguo376/project/model/MMCA/chartsumm_entail_singal/test_s_output.json"
]
chart_root = "/data/jguo376/project/dataset/chartsumm/chart_images/"
max_items = None  # None 表示全部处理

# ========= 加载模型 =========
print("📦 加载基础模型权重...")
model = MplugOwlForConditionalGeneration.from_pretrained(
    model_path_for_weights,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

if lora_path:
    print(f"🪄 加载 LoRA adapter: {lora_path}")
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


# ========= 推理函数 =========
def run_inference(input_json, output_json, chart_root, max_items=None):
    print(f"\n📂 处理文件：{input_json}")
    
    # 已处理断点信息
    if os.path.exists(output_json):
        with open(output_json, "r", encoding="utf-8") as f:
            try:
                existing_results = json.load(f)
            except json.JSONDecodeError:
                existing_results = []
        processed_imgs = set(item["image"] for item in existing_results)
    else:
        existing_results = []
        processed_imgs = set()

    results = existing_results.copy()

    with open(input_json, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    if max_items is not None:
        data_list = data_list[:max_items]

    for idx, item in enumerate(tqdm(data_list, desc=f"Generating captions for {os.path.basename(input_json)}")):
        image_name = item.get("image") or item.get("img")  # 两种可能字段名
        img_path = os.path.join(chart_root, image_name)

        if image_name in processed_imgs:
            continue
        if not os.path.exists(img_path):
            print(f"⚠️ 图像不存在: {img_path}")
            continue

        try:
            image = Image.open(img_path).convert("RGB")
            inputs = processor(text=[query_prompt], images=[image], return_tensors="pt")
            inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                output = model.generate(**inputs, **generate_kwargs)
                caption = tokenizer.decode(output[0], skip_special_tokens=True)
        except Exception as e:
            print(f"❌ 错误：{image_name} - {e}")
            traceback.print_exc()
            caption = f"[ERROR] {str(e)}"

        result = {
            "image": image_name,
            "generated_caption": caption
        }
        results.append(result)
        processed_imgs.add(image_name)

        print(f"[✓] {image_name}")
        print(f"    → {caption}\n")

        # 中间保存
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ 完成：{output_json}，共生成 {len(results)} 条描述")

# ========= 批量处理两个文件 =========
for in_path, out_path in zip(input_jsons, output_jsons):
    run_inference(in_path, out_path, chart_root, max_items=max_items)
