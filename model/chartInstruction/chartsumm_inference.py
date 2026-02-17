import os
import json
from tqdm import tqdm
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoImageProcessor, AutoModelForSeq2SeqLM

# ===== 模型配置 =====
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
model_id = "ahmed-masry/ChartInstruct-FlanT5-XL"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ 使用设备: {device}")

# ===== 多文件路径配置 =====
input_jsons = [
    "/data/jguo376/project/dataset/chartsumm/test_k.json",
    "/data/jguo376/project/dataset/chartsumm/test_s.json"
]
output_jsons = [
    "/data/jguo376/project/model/chartInstruction/test_k_output.json",
    "/data/jguo376/project/model/chartInstruction/test_s_output.json"
]
image_root = "/data/jguo376/project/dataset/chartsumm/chart_images"

# ===== 加载模型 =====
print("🚀 加载 ChartInstruct-FlanT5-XL 模型中...")
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    trust_remote_code=True
).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
image_processor = AutoImageProcessor.from_pretrained(model_id)

# ===== 推理函数 =====
def infer_chartinstruct(image_path, input_question="Please describe the chart."):
    image = Image.open(image_path).convert("RGB")
    prompt = f"<image>\n Question: {input_question} Answer: "

    text_inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512
    )
    pixel_values = image_processor(images=image, return_tensors="pt")["pixel_values"].to(
        device, dtype=torch.float16 if device.type == "cuda" else torch.float32
    )
    print(f"[INFO] pixel_values.shape for {os.path.basename(image_path)}: {pixel_values.shape}")

    inputs = {
        "input_ids": text_inputs["input_ids"].to(device),
        "attention_mask": text_inputs["attention_mask"].to(device),
        "pixel_values": pixel_values
    }

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            num_beams=4,
            max_new_tokens=512,
            early_stopping=True
        )

    output_text = tokenizer.batch_decode(
        outputs,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )[0]
    return output_text

# ===== 批量处理每个文件 =====
for input_path, output_path in zip(input_jsons, output_jsons):
    print(f"\n📂 正在处理文件: {input_path}")
    
    with open(input_path, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    # ==== 加载已完成项（用于断点续跑） ====
    done_images = set()
    results = []
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            results = json.load(f)
            done_images = {item["image"] for item in results}
        print(f"🔄 已检测到 {len(done_images)} 条结果，将跳过已完成的图像。")

    # ==== 开始处理 ====
    for idx, item in enumerate(tqdm(data_list, desc=f"Generating captions for {os.path.basename(input_path)}")):
        image_name = item.get("image") or item.get("img")
        if image_name in done_images:
            continue

        image_path = os.path.join(image_root, image_name)
        if not os.path.exists(image_path):
            caption = f"[ERROR] Image not found: {image_path}"
        else:
            try:
                caption = infer_chartinstruct(image_path)
            except Exception as e:
                caption = f"[ERROR] {str(e)}"

        result_item = {
            "image": image_name,
            "generated_caption": caption
        }
        results.append(result_item)

        print(f"[✓] {image_name}\n    → {caption}\n")

        # 每处理10条保存一次中间结果
        if len(results) % 10 == 0:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    # 最终保存完整结果
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✅ 最终结果保存到: {output_path}")
