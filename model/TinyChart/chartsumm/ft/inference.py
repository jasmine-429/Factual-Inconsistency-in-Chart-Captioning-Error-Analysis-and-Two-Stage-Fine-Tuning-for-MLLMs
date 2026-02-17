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
lora_path = "/data/jguo376/project/model/TinyChart/checkpoints/chartsumm_caption/checkpoint-1400"

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



# ========= 输入文件配置 =========
input_json_paths = [
    ("valid_k", "/data/jguo376/project/dataset/chartsumm/valid_k.json"),
    ("valid_s", "/data/jguo376/project/dataset/chartsumm/valid_s.json")
]
chart_image_root = "/data/jguo376/project/dataset/chartsumm/chart_images"
output_dir = "/data/jguo376/project/model/TinyChart/chartsumm/ft/vail"
os.makedirs(output_dir, exist_ok=True)

prompt = "Please describe the chart."
save_every = 20
max_test = None

# ========= 遍历每个数据集 =========
for tag, json_path in input_json_paths:
    print(f"\n📂 Processing dataset: {tag}")
    output_json = os.path.join(output_dir, f"chartsumm_{tag}.json")

    # === 加载数据 ===
    with open(json_path, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    # === 加载已完成项（断点续跑）===
    processed_imgs = set()
    if os.path.exists(output_json):
        with open(output_json, "r", encoding="utf-8") as f:
            try:
                existing_results = json.load(f)
                for entry in existing_results:
                    processed_imgs.add(entry["image"])
            except Exception:
                existing_results = []
    else:
        existing_results = []

    results = existing_results.copy()
    count = 0

    # === 推理开始 ===
    model.eval()
    torch.set_grad_enabled(False)
    for item in tqdm(data_list, desc=f"Generating captions for {tag}"):
        image_name = item["image"]
        if image_name in processed_imgs:
            continue

        image_path = os.path.join(chart_image_root, image_name)

        if not os.path.exists(image_path):
            caption = f"[ERROR] Image not found: {image_path}"
        else:
            try:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    caption = inference_model(
                        [image_path],
                        prompt,
                        model,
                        tokenizer,
                        image_processor,
                        context_len,
                        conv_mode="phi",
                        max_new_tokens=512
                    )
            except Exception as e:
                caption = f"[ERROR] {str(e)}"

        results.append({
            "image": image_name,
            "generated_caption": caption
        })
        processed_imgs.add(image_name)
        count += 1

        print(f"[✓] {image_name}")
        print(f"    → {caption}\n")

        if save_every and count % save_every == 0:
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

        if max_test is not None and count >= max_test:
            break

    # === 最终保存 ===
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ Finished: {tag} → {output_json}")
