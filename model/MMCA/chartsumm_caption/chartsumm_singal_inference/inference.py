import os
import sys

# ===== 设定环境 =====

sys.path.append("/data/jguo376/project/model/mPLUG-Owl/mPLUG-Owl")

import json
from PIL import Image
from tqdm import tqdm
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer
from mplug_owl.modeling_mplug_owl import MplugOwlForConditionalGeneration
from mplug_owl.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor

# ===== 路径配置 =====
base_model_path = "/data/jguo376/pretrained_models/mplug-owl-llama-7b"
lora_path = "/data/jguo376/pretrained_models/MMCA/mmca_lora_weights.bin"
input_jsonl = "/data/jguo376/project/dataset/chartsumm/test_s.json"
output_json = "/data/jguo376/project/model/MMCA/chartsumm_caption/chartsumm_singal_inference/test_s_outpu.json"
chart_root = "/data/jguo376/project/dataset/chartsumm/chart_images/"


# ===== 加载模型与 LoRA 权重 =====
print("📦 加载模型与 LoRA 权重...")
model = MplugOwlForConditionalGeneration.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
peft_config = LoraConfig(
    target_modules=r'.*language_model.*\.(q_proj|v_proj)',
    inference_mode=True,
    r=8,
    lora_alpha=32,
    lora_dropout=0.05
)
model = get_peft_model(model, peft_config)
lora_weights = torch.load(lora_path, map_location="cpu", weights_only=True)

model.load_state_dict(lora_weights, strict=False)

# ===== 加载处理器 =====
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
image_processor = MplugOwlImageProcessor.from_pretrained(base_model_path)
processor = MplugOwlProcessor(image_processor, tokenizer)

# ===== 推理参数 =====
generate_kwargs = {
    'do_sample': True,
    'top_k': 5,
    'max_length': 256
}
query_prompt = """The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
Human: <image>
Human: Provide a short analytical description of the chart based on the data it shows.
AI:"""

# ===== 加载输入数据 =====
with open(input_jsonl, "r", encoding="utf-8") as f:
    data_list = json.load(f)

results = []
max_test = None  # 设置为数字限制条数，如 5；None 表示全部处理

# ===== 批量处理图表图像 =====
print("🚀 开始批量生成图表描述...")
count = 0
for item in tqdm(data_list, desc="Generating captions"):
    if max_test is not None and count >= max_test:
        break


    imgname = item.get("image") or item.get("img")  # 两种可能字段名
    image_path = os.path.join(chart_root, imgname)

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

    item["model_name"] = "mPLUG-Owl + MMCA"
    item["img"] = image_path  # 使用绝对路径
    item["generated_caption"] = caption
    results.append(item)
    count += 1

    print(f"[✓] {imgname}")
    print(f"    → {caption}\n")

# ===== 保存输出文件 =====
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n✅ 推理完成，共生成 {len(results)} 条图表描述，输出保存至：{output_json}")
