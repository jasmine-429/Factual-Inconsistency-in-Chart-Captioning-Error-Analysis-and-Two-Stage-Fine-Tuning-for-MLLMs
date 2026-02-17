from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
from PIL import Image
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# ✅ 加载模型
model_id = "google/pix2struct-chartqa-base"
processor = Pix2StructProcessor.from_pretrained(model_id)
model = Pix2StructForConditionalGeneration.from_pretrained(model_id).to("cuda").eval()

# ✅ 读取图像
image_path = "/data/jguo376/project/dataset/ChartX_dataset/ChartX/bar_chart/png/bar_1.png"
image = Image.open(image_path).convert("RGB")

# ✅ 添加 “问题 prompt” 才能生成结果
inputs = processor(images=image, text="Provide a short analytical description of the chart based on the data it shows.", return_tensors="pt").to("cuda")

# ✅ 推理生成 caption
outputs = model.generate(**inputs, max_new_tokens=64)
caption = processor.decode(outputs[0], skip_special_tokens=True)

print("📊 Chart Caption:\n", caption)
