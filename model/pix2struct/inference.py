from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
from PIL import Image
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

model_id = "google/pix2struct-textcaps-base"

print("📦 Loading processor and model...")
processor = Pix2StructProcessor.from_pretrained(model_id)
model = Pix2StructForConditionalGeneration.from_pretrained(model_id).to("cuda").eval()

image_path = "/data/jguo376/project/dataset/ChartX_dataset/ChartX/bar_chart/png/bar_5.png"
image = Image.open(image_path).convert("RGB")

inputs = processor(images=image, return_tensors="pt").to("cuda")

print("🧠 Generating caption...")
outputs = model.generate(**inputs, max_new_tokens=64)
caption = processor.decode(outputs[0], skip_special_tokens=True)

print("📊 Chart Caption:\n", caption)
