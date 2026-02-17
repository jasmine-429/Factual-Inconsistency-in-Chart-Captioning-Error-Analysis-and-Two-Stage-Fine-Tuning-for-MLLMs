import torch
from PIL import Image
from transformers import AutoTokenizer, AutoImageProcessor, AutoModelForSeq2SeqLM
import os

# ===== 模型配置 =====
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# ========= 配置路径 =========
image_path = "/data/jguo376/project/dataset/ChartX_dataset/ChartX/bar_chart/png/bar_1.png"
input_question = "Please describe the chart."

# ========= 设备选择 =========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ 使用设备: {device}")

# ========= 加载模型和处理器（手动方式）=========
model_id = "ahmed-masry/ChartInstruct-FlanT5-XL"

model = AutoModelForSeq2SeqLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    trust_remote_code=True
).to(device)

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
image_processor = AutoImageProcessor.from_pretrained(model_id)

# ========= 读取图片和构建输入 =========
image = Image.open(image_path).convert("RGB")
input_prompt = f"<image>\n Question: {input_question} Answer: "

# 文本编码
text_inputs = tokenizer(
    input_prompt,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=512
)

# 图像编码并转为指定设备
pixel_values = image_processor(images=image, return_tensors="pt")["pixel_values"].to(
    device, dtype=torch.float16 if device.type == "cuda" else torch.float32
)

# 合并为模型输入
inputs = {
    "input_ids": text_inputs["input_ids"].to(device),
    "attention_mask": text_inputs["attention_mask"].to(device),
    "pixel_values": pixel_values
}

# ========= 推理并生成答案 =========
with torch.no_grad():
    generate_ids = model.generate(
        **inputs,
        num_beams=4,
        max_new_tokens=512,
        early_stopping=True
    )

# ========= 解码输出 =========
output = tokenizer.batch_decode(
    generate_ids,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=True
)[0]

# ========= 打印结果 =========
print("\n📊 输入问题:", input_question)
print("🤖 模型回答:", output)
