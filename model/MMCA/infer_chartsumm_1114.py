import os
import sys
import json
from PIL import Image
import torch
from tqdm import tqdm

# ======= 配置路径 =======
BASE_MODEL = "/data/jguo376/pretrained_models/mplug-owl-llama-7b"
LORA_PATH = "/data/jguo376/pretrained_models/MMCA/mmca_lora_weights.bin"
IMAGE_ROOT = "/data/jguo376/project/dataset/chartsumm/chart_images"
INPUT_JSON = "/data/jguo376/project/dataset/chartsumm/test_s.json"  # 你的输入文件
OUTPUT_JSONL = "/data/jguo376/project/model/mmca_caption/captions_mmca_s.jsonl"    # 输出文件

# ======= 断点续跑：加载已完成记录 =======
done_images = set()
if os.path.exists(OUTPUT_JSONL):
    with open(OUTPUT_JSONL, "r") as f:
        for line in f:
            try:
                item = json.loads(line)
                done_images.add(item["image"])
            except:
                continue

print(f"🔁 已完成 {len(done_images)} 条，将跳过这些样本")

# ======= 添加 mPLUG-Owl 源码路径 =======
sys.path.append("/data/jguo376/project/model/mPLUG-Owl/mPLUG-Owl")

# ======= 模型加载 =======
print("📦 加载模型...")
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer
from mplug_owl.modeling_mplug_owl import MplugOwlForConditionalGeneration
from mplug_owl.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor

model = MplugOwlForConditionalGeneration.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

peft_cfg = LoraConfig(
    target_modules=r'.*language_model.*\.(q_proj|v_proj)',
    inference_mode=True,
    r=8,
    lora_alpha=32,
    lora_dropout=0.05
)
model = get_peft_model(model, peft_cfg)

print("📥 加载 LoRA 权重...")
lora_weights = torch.load(LORA_PATH, map_location="cpu")
model.load_state_dict(lora_weights, strict=False)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
image_processor = MplugOwlImageProcessor.from_pretrained(BASE_MODEL)
processor = MplugOwlProcessor(image_processor, tokenizer)

# ======= 生成 Prompt =======
PROMPT = """The following is a conversation between a curious human and AI assistant. 
The assistant gives helpful, detailed, and polite answers.
Human: <image>
Human: Provide a short analytical description of the chart based on the data it shows.
AI:"""

# ======= 读取数据 =======
with open(INPUT_JSON, "r") as f:
    dataset = json.load(f)

# ======= 推理参数 =======
GEN_KWARGS = {
    "do_sample": True,
    "top_k": 5,
    "max_length": 512,
}

# ======= 开始推理 =======
print("🚀 开始批量生成 caption...")

with open(OUTPUT_JSONL, "a") as fout:
    for item in tqdm(dataset):
        image_name = item["image"]

        # 断点续跑
        if image_name in done_images:
            continue

        image_path = os.path.join(IMAGE_ROOT, image_name)
        if not os.path.exists(image_path):
            print(f"❌ 找不到图片: {image_path}")
            continue

        # 读取图片
        image = Image.open(image_path).convert("RGB")

        # 构造输入
        inputs = processor(text=[PROMPT], images=[image], return_tensors="pt")
        inputs = {k: (v.bfloat16() if v.dtype == torch.float else v).to(model.device) for k, v in inputs.items()}

        # 生成
        with torch.no_grad():
            output = model.generate(**inputs, **GEN_KWARGS)
            caption = tokenizer.decode(output[0], skip_special_tokens=True)

        # 写出
        result = {"image": image_name, "caption": caption}
        fout.write(json.dumps(result, ensure_ascii=False) + "\n")

        # 实时 flush，便于断点续跑
        fout.flush()

print("🎉 全部完成！")
