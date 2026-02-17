from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
from PIL import Image
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# 1. 模型 ID
model_id = "aravind-selvam/pix2struct_chart"

# 2. 加载模型和处理器
print("📦 Loading model and processor...")
processor = Pix2StructProcessor.from_pretrained(model_id)
model = Pix2StructForConditionalGeneration.from_pretrained(model_id).to("cuda").eval()

# 3. 加载图像（图表）
image_path = "/data/jguo376/project/dataset/ChartX_dataset/ChartX/bar_chart/png/bar_1.png"  # ✅ 替换为你的图表图像路径
image = Image.open(image_path).convert("RGB")

# 4. 预处理并生成 caption
inputs = processor(images=image, return_tensors="pt").to("cuda")

print("🧠 Generating caption...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=64,         # 生成最大 token 数
        num_beams=4,               # Beam search
        temperature=0.7,           # 控制多样性
        top_p=0.9
    )

# 5. 解码输出
caption = processor.decode(outputs[0], skip_special_tokens=True)
print("\n📊 Chart Caption:\n", caption)
