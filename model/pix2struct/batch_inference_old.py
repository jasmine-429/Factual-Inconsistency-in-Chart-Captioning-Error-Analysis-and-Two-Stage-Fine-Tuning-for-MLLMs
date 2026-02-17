import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration

# ===== 模型配置 =====
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
model_id = "oroikon/ft_pix2struct_chart_captioning"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("📦 Loading model and processor...")
processor = Pix2StructProcessor.from_pretrained(model_id)
model = Pix2StructForConditionalGeneration.from_pretrained(model_id).to(device).eval()

# ===== 推理函数 =====
def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption.replace("\x0A", "").strip()

# ===== 路径配置 =====
dataset_root = "/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/all_type/caption/deep_analysis/caption_all"
input_jsonl = os.path.join(dataset_root, "chartx_selected_fields.json")
output_json = os.path.join(dataset_root, "pix2struct_caption_output.json")
chart_root = "/data/jguo376/project/dataset/ChartX_dataset/ChartX"


# ===== 读取 JSONL 输入文件 =====
with open(input_jsonl, "r", encoding="utf-8") as f:
    data_list = json.load(f)
    #data_list = [json.loads(line) for line in f.readlines()]

results = []
max_test = 4  # 改为整数以限制数量，如 10；否则设为 None

# ===== 批量处理 =====
print("🚀 Start caption generation...")
count = 0
for item in tqdm(data_list, desc="Generating captions"):
    if max_test is not None and count >= max_test:
        break

    imgname = item["imgname"]
    rel_path = item["img"]
    image_path = os.path.join(chart_root, rel_path.replace("./", ""))

    if not os.path.exists(image_path):
        caption = f"[ERROR] Image not found: {image_path}"
    else:
        try:
            caption = generate_caption(image_path)
        except Exception as e:
            caption = f"[ERROR] {str(e)}"

    item["model_name"] = model_id
    item["img"] = image_path
    item["generated_caption"] = caption
    results.append(item)
    count += 1

    print(f"[✓] {imgname}")
    print(f"    → {caption}\n")

# ===== 写入输出 JSON 文件 =====
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n✅ Done! Total: {len(results)} captions saved to: {output_json}")
