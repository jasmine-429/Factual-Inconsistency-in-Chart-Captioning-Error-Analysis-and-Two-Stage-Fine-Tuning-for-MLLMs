from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
from PIL import Image
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# ===== 配置模型和图像路径 =====
model_id = "google/matcha-chart2text-statista"  # 或者 matcha-chart2text-pew
image_path = "/data/jguo376/project/dataset/ChartX_dataset/ChartX/bar_chart/png/bar_158.png"
query = "Provide a short analytical description of the chart based on the data it shows."  # 任务指令

# ===== 加载模型与处理器 =====
processor = Pix2StructProcessor.from_pretrained(model_id)
model = Pix2StructForConditionalGeneration.from_pretrained(model_id)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# ===== 加载图像并生成输入 =====
image = Image.open(image_path).convert("RGB")
inputs = processor(images=image, text=query, return_tensors="pt").to(device)

# ===== 生成描述 =====
outputs = model.generate(**inputs, max_new_tokens=128)
caption = processor.batch_decode(outputs, skip_special_tokens=True)[0]

print("📝 Chart Caption:", caption)
