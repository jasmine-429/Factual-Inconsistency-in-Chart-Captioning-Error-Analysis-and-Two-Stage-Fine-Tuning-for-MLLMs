from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
from PIL import Image
import torch

# 1. 模型名称
model_id = "oroikon/ft_pix2struct_chart_captioning"

# 2. 加载模型和 processor
print("📦 Loading model and processor...")
processor = Pix2StructProcessor.from_pretrained(model_id)
model = Pix2StructForConditionalGeneration.from_pretrained(model_id).to("cuda").eval()

# 3. 加载图像
image_path = "/data/jguo376/project/dataset/ChartX_dataset/ChartX/bar_chart/png/bar_1.png"  # ✅ 替换为你的图表图片路径
image = Image.open(image_path).convert("RGB")

# 4. 图像预处理
inputs = processor(images=image, return_tensors="pt").to("cuda")

# 5. 推理生成
print("🧠 Generating caption...")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=64)

# 6. 解码输出
caption = processor.decode(outputs[0], skip_special_tokens=True)
print("\n📊 Generated Chart Caption:\n", caption)
