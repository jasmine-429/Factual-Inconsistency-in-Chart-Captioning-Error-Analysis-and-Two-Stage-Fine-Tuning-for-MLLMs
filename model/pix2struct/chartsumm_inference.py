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

# ===== 数据路径配置 =====
image_root = "/data/jguo376/project/dataset/chartsumm/chart_images"
input_jsons = [
    "/data/jguo376/project/dataset/chartsumm/test_k.json",
    "/data/jguo376/project/dataset/chartsumm/test_s.json"
]
output_jsons = [
    "/data/jguo376/project/model/pix2struct/test_k_output.json",
    "/data/jguo376/project/model/pix2struct/test_s_output.json"
]

# ===== 主循环：处理多个文件 =====
for input_json, output_json in zip(input_jsons, output_jsons):
    print(f"\n🚀 Start caption generation for: {input_json}")

    # 加载数据
    with open(input_json, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    # 读取已有结果（断点续跑）
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
    max_test = None
    save_every = 20
    count = 0

    for item in tqdm(data_list, desc=f"Generating captions for {os.path.basename(input_json)}"):
        image_name = item["image"]
        image_path = os.path.join(image_root, image_name)

        if image_name in processed_imgs:
            continue
        if max_test is not None and count >= max_test:
            break

        # 推理
        if not os.path.exists(image_path):
            caption = f"[ERROR] Image not found: {image_path}"
        else:
            try:
                caption = generate_caption(image_path)
            except Exception as e:
                caption = f"[ERROR] {str(e)}"

        output = {
            "image": image_name,
            "generated_caption": caption
        }

        results.append(output)
        processed_imgs.add(image_name)
        count += 1

        print(f"[✓] {image_name}")
        print(f"    → {caption}\n")

        # 中间保存
        if count % save_every == 0:
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

        torch.cuda.empty_cache()

    # 最终保存
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Done! Total: {len(results)} captions saved to: {output_json}")
