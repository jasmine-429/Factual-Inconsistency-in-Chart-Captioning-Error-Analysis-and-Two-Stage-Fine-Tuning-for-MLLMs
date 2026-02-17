import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration

# ===== 模型配置 =====
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
model_id = "google/matcha-chart2text-statista"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = Pix2StructProcessor.from_pretrained(model_id)
model = Pix2StructForConditionalGeneration.from_pretrained(model_id).to(device).eval()
query = "Provide a short analytical description of the chart based on the data it shows."

# ===== 路径配置 =====
Img_root = "/data/jguo376/project/dataset/ChartX_dataset/ChartX"
dataset_root = "/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/all_type/caption/deep_analysis/caption_all"
input_json = os.path.join(dataset_root, "chartx_selected_fields.json")
output_json = os.path.join(dataset_root, "matcha_caption_output_sta333.json")

# ===== 控制处理条数 =====
max_test = 4  # None 表示不限制处理条数

# ===== 加载输入数据 =====
with open(input_json, "r", encoding="utf-8") as f:
    data_list = json.load(f)

# ===== 加载已存在的最终输出（支持断点续跑）=====
existing_data = []
processed_imgs = set()
if os.path.exists(output_json):
    with open(output_json, "r", encoding="utf-8") as f:
        try:
            existing_data = json.load(f)
            processed_imgs = set(item["img"] for item in existing_data)
        except:
            existing_data = []
            processed_imgs = set()

# ===== 推理与保存 =====
save_every = 20
batch = []
new_count = 0

for entry in tqdm(data_list):
    if max_test is not None and new_count >= max_test:
        break

    img_name = entry["img"]
    if img_name in processed_imgs:
        continue

    img_path = os.path.join(Img_root, img_name)
    try:
        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, text=query, return_tensors="pt").to(device)
        predictions = model.generate(**inputs, max_new_tokens=512)
        caption = processor.decode(predictions[0], skip_special_tokens=True)
    except Exception as e:
        print(f"[ERROR] {img_name}: {e}")
        continue

    entry["model_name"] = model_id
    entry["generated_caption"] = caption

    batch.append(entry)
    new_count += 1

    # 每 20 条写一次完整 JSON 数组
    if len(batch) >= save_every:
        existing_data.extend(batch)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)
        processed_imgs.update(item["img"] for item in batch)
        batch = []

# 保存最后剩余条目
if batch:
    existing_data.extend(batch)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=2)

print(f"✅ 完成，总共保存了 {len(existing_data)} 条样本到 {output_json}")
