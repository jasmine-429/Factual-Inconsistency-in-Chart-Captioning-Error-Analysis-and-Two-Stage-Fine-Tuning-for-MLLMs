import os
import sys

# ===== 设定只使用 GPU 4 =====

# ===== 添加 mPLUG-Owl 源码路径 =====
sys.path.append("/data/jguo376/project/model/mPLUG-Owl/mPLUG-Owl")

# ===== 导入依赖 =====
import torch
from PIL import Image
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer
from mplug_owl.modeling_mplug_owl import MplugOwlForConditionalGeneration
from mplug_owl.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor

# ===== 路径配置 =====
base_model_path = "/data/jguo376/pretrained_models/mplug-owl-llama-7b"
lora_path = "/data/jguo376/pretrained_models/MMCA/mmca_lora_weights.bin"
image_path = "/data/jguo376/project/dataset/chartsumm/chart_images/test_k_2.png"

# ===== 加载模型 & LoRA 权重 =====
print("📦 加载模型与 LoRA 权重...")
model = MplugOwlForConditionalGeneration.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
peft_config = LoraConfig(
    target_modules=r'.*language_model.*\.(q_proj|v_proj)',
    inference_mode=True,
    r=8,
    lora_alpha=32,
    lora_dropout=0.05
)
model = get_peft_model(model, peft_config)
lora_weights = torch.load(lora_path, map_location="cpu")
model.load_state_dict(lora_weights, strict=False)

# ===== 加载处理器 =====
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
image_processor = MplugOwlImageProcessor.from_pretrained(base_model_path)
processor = MplugOwlProcessor(image_processor, tokenizer)

# ===== 构造对话 Prompt（支持复杂推理）=====
prompt = """The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
Human: <image>
Human: Provide a short analytical description of the chart based on the data it shows.
AI:"""

# ===== 读取图像 & 构造输入 =====
image = Image.open(image_path).convert("RGB")
inputs = processor(text=[prompt], images=[image], return_tensors="pt")
inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
inputs = {k: v.to(model.device) for k, v in inputs.items()}

# ===== 生成输出 =====
generate_kwargs = {
    'do_sample': True,
    'top_k': 5,
    'max_length': 512
}
print("🧠 正在生成图表描述...")
with torch.no_grad():
    output = model.generate(**inputs, **generate_kwargs)
    caption = tokenizer.decode(output[0], skip_special_tokens=True)

# ===== 输出结果 =====
print("\n📊 图表描述结果：")
print(caption)