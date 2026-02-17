import os
import json
from tqdm import tqdm
from PIL import Image

import sys
sys.path.append("/data/jguo376/project/model/TinyChart")

import torch
from tinychart.model.builder import load_pretrained_model
from tinychart.mm_utils import get_model_name_from_path
from tinychart.eval.run_tiny_chart import inference_model

# ========= 环境配置 =========
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
device = "cuda:0"

# ========= 模型路径 =========
model_path = "/data/jguo376/pretrained_models/TinyChart-3B-768"
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path, 
    model_base=None,
    model_name=get_model_name_from_path(model_path),
    device=device
)
model = model.half()

# ========= 输入文件配置 =========
input_json_paths = [
    ("test_k", "/data/jguo376/project/dataset/chartsumm/test_k.json"),
    ("test_s", "/data/jguo376/project/dataset/chartsumm/test_s.json")
]
chart_image_root = "/data/jguo376/project/dataset/chartsumm/chart_images"
output_dir = "/data/jguo376/project/model/TinyChart/chartsumm/org/test"
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
