from PIL import Image
from transformers import AutoTokenizer, AutoImageProcessor, AutoModelForSeq2SeqLM
import torch

# 模型 ID
model_id = "ahmed-masry/ChartInstruct-FlanT5-XL"

# 设置设备
device = torch.device("cuda")

# 加载 tokenizer（禁用 fast 模式避免错误）
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
processor = AutoImageProcessor.from_pretrained(model_id)

# 加载图像
image_path = "/data/jguo376/project/dataset/ChartX_dataset/ChartX/bar_chart/png/bar_1.png"
image = Image.open(image_path).convert("RGB")

# 处理图像
pixel_values = processor(images=image, return_tensors="pt")["pixel_values"].to(device)

# 打印 shape
print(f"✅ pixel_values.shape: {pixel_values.shape}")
