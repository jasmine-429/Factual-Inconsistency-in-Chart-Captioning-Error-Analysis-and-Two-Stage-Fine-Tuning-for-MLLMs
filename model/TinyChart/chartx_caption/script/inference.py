import sys
sys.path.append("/data/jguo376/project/model/TinyChart")

import torch
import json
import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.stats import kendalltau
from tinychart.model.builder import load_pretrained_model
from tinychart.mm_utils import get_model_name_from_path
from tinychart.eval.run_tiny_chart import inference_model
from peft import PeftModel
# ========= 环境配置 =========
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
model_path = "/data/jguo376/pretrained_models/TinyChart-3B-768"
use_lora = True  # 修改为 True
lora_path = "/data/jguo376/project/model/TinyChart/checkpoints/chartx_caption/checkpoint-800"

# ========= 加载模型 =========
print("Loading model...")
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path,
    model_base=None,  # 设为 None
    model_name=get_model_name_from_path(model_path),
    device="cuda:0"
)

if use_lora:
    print("Loading LoRA...")
    model = PeftModel.from_pretrained(model, lora_path)
    print("Merging LoRA weights...")
    model = model.merge_and_unload()
    print("LoRA weights loaded and merged.")

# 关键修复：统一数据类型
model = model.half()  # 而不是 model.float()
print(f"Model loaded on device: {next(model.parameters()).device}")
print(f"Model dtype: {next(model.parameters()).dtype}")


# ========= 路径配置 =========
dataset_root = "/data/jguo376/project/dataset/test_dataset/ChartX/test_eva_data/data"
output_root = "/data/jguo376/project/model/TinyChart/chartx_caption"
input_jsonl = os.path.join(dataset_root, "eva_test.json")
output_json = os.path.join(output_root, "ft_output.json")
chart_root = "/data/jguo376/project/dataset/ChartX_dataset/ChartX"
prompt = "Please describe the chart."
# ========= 加载输入数据 =========
with open(input_jsonl, "r", encoding="utf-8") as f:
    data_list = json.load(f)

# ========= 断点续跑支持：记录已处理图片 =========
processed_imgs = set()
if os.path.exists(output_json):
    with open(output_json, "r", encoding="utf-8") as f:
        try:
            existing_results = json.load(f)
            for entry in existing_results:
                processed_imgs.add(entry["img"])  # 使用绝对路径作为唯一标识
        except Exception:
            existing_results = []
else:
    existing_results = []

results = existing_results.copy()

# ========= 参数配置 =========
max_test = None      # 若只跑部分，可设置为数字，如 10；否则设为 None
save_every = 20      # 每处理 N 张图片保存一次结果
count = 0

# ========= 开始处理 =========
print("🚀 Start TinyChart caption generation...")
for item in tqdm(data_list, desc="Generating captions"):
    rel_path = item["img"]
    image_path = os.path.join(chart_root, rel_path.replace("./", ""))

    if image_path in processed_imgs:
        continue

    if max_test is not None and count >= max_test:
        break

    if not os.path.exists(image_path):
        caption = f"[ERROR] Image not found: {image_path}"
    else:
        try:
            caption = inference_model([image_path], prompt, model, tokenizer, image_processor, context_len, conv_mode="phi", max_new_tokens=512)
        except Exception as e:
            caption = f"[ERROR] {str(e)}"

    item["model_name"] = model_path
    item["img"] = image_path  # 绝对路径
    item["generated_caption"] = caption
    results.append(item)
    processed_imgs.add(image_path)
    count += 1

    print(f"[✓] {item.get('imgname')}")
    print(f"    → {caption}\n")

    # ========= 中间保存 =========
    if count % save_every == 0:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

# ========= 最终保存 =========
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n✅ TinyChart 推理完成！共处理 {len(results)} 张图，输出保存至: {output_json}")
