import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from PIL import Image

# 关闭梯度以节省显存
torch.set_grad_enabled(False)

# 模型路径（HuggingFace）
ckpt_path = "internlm/internlm-xcomposer2d5-7b"

# 加载配置并强制使用普通 attention
config = AutoConfig.from_pretrained(ckpt_path, trust_remote_code=True)
config.attn_implementation = "eager"  # 🚫 不使用 flash attention

# 加载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
model = AutoModel.from_pretrained(
    ckpt_path,
    config=config,  # ✅ 使用修改后的 config
    torch_dtype=torch.float16,
    trust_remote_code=True
).cuda().eval()
model.tokenizer = tokenizer  # 显式绑定 tokenizer

# 图像路径和文本 prompt
query = "Write a concise paragraph that describes the chart, including key values, categories, and noticeable trends."
def ensure_rgb(image_path):
    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
        img.save(image_path)  # 或另存为新的路径
    return image_path

image_path = ensure_rgb("/data/jguo376/project/dataset/ChartX_dataset/ChartX/bar_chart/png/bar_5.png")
image = [image_path]

# 推理
with torch.autocast(device_type='cuda', dtype=torch.float16):
    response, _ = model.chat(
        tokenizer,
        query,
        image,
        do_sample=False,
        num_beams=3,
        use_meta=True
    )

print("📊 图表描述结果：", response)
