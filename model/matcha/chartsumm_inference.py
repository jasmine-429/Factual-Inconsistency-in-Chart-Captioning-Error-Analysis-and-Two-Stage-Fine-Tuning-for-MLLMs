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

print("📦 Loading model and processor...")
processor = Pix2StructProcessor.from_pretrained(model_id)
model = Pix2StructForConditionalGeneration.from_pretrained(model_id).to(device).eval()
query = "Please describe the chart."

# ===== 路径配置（chartsumm 数据）=====
image_root = "/data/jguo376/project/dataset/chartsumm/chart_images"
input_jsons = [
    "/data/jguo376/project/dataset/chartsumm/test_k.json",
    "/data/jguo376/project/dataset/chartsumm/test_s.json"
]
output_jsons = [
    "/data/jguo376/project/model/matcha/test_k_output.json",
    "/data/jguo376/project/model/matcha/test_s_output.json"
]

# ===== 控制参数 =====
max_test = None   # 设为整数仅处理前N条；None表示处理全部
save_every = 20   # 每处理N条保存一次（增量保存）

# ===== 推理函数 =====
def generate_caption(image_path: str, query_text: str) -> str:
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, text=query_text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512)
    caption = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return caption.replace("\x0A", "").strip()

# ===== 主流程：逐个文件处理 =====
for input_json, output_json in zip(input_jsons, output_jsons):
    print(f"\n🚀 Start caption generation for: {input_json}")

    # 读取输入
    with open(input_json, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    if max_test is not None:
        data_list = data_list[:max_test]

    # 断点续跑：读取已完成结果（仅两字段）
    processed_imgs = set()
    results = []
    if os.path.exists(output_json):
        with open(output_json, "r", encoding="utf-8") as f:
            try:
                results = json.load(f)
                for entry in results:
                    # 已有输出中的 image 字段作为唯一键
                    processed_imgs.add(entry.get("image"))
                print(f"🔁 Loaded {len(results)} existing results (resume enabled)")
            except Exception:
                # 若老文件非期望结构，则从空开始
                results = []
                processed_imgs = set()

    new_buffer = []

    # 推理
    pbar = tqdm(data_list, desc=f"Generating captions for {os.path.basename(input_json)}")
    for item in pbar:
        image_name = item.get("image")
        if not image_name:
            # 缺少 image 字段则跳过
            continue

        if image_name in processed_imgs:
            # 已处理过，跳过
            continue

        image_path = os.path.join(image_root, image_name)

        # 生成 caption
        if not os.path.exists(image_path):
            caption = f"[ERROR] Image not found: {image_path}"
        else:
            try:
                caption = generate_caption(image_path, query)
            except Exception as e:
                caption = f"[ERROR] {str(e)}"

        # 只保留两字段
        out_rec = {
            "image": image_name,
            "generated_caption": caption
        }
        results.append(out_rec)
        new_buffer.append(out_rec)
        processed_imgs.add(image_name)

        # 增量保存
        if len(new_buffer) >= save_every:
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            new_buffer.clear()

    # 最终保存
    if new_buffer:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ Done! Total: {len(results)} captions saved to: {output_json}")
