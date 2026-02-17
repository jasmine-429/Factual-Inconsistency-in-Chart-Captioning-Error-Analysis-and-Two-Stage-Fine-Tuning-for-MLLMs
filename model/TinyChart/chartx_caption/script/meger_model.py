import sys
sys.path.append("/data/jguo376/project/model/TinyChart")

import torch
import json
import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.stats import kendalltau
from tinychart.model.builder import load_pretrained_model
from tinychart.mm_utils import get_model_name_from_path
from tinychart.eval.run_tiny_chart import inference_model
from peft import PeftModel

# ======= 配置路径 =======
model_path = "/data/jguo376/pretrained_models/TinyChart-3B-768"
lora_paths = [
    "/data/jguo376/project/model/TinyChart/checkpoints/chartx_caption/checkpoint-800",
    "/data/jguo376/project/model/TinyChart/checkpoints/chart_entail/checkpoint-700"
]
save_path = "/data/jguo376/project/model/TinyChart/merged_models/chart_entail_caption_merged"

device = "cuda:0"

# ======= 加载 base 模型 =======
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path),
    device=device
)
model = model.half()

# ======= 依次合并多个 LoRA =======
for i, lora_path in enumerate(lora_paths):
    print(f"🔄 Loading & merging LoRA {i+1}: {lora_path}")
    model = PeftModel.from_pretrained(model, lora_path)
    model = model.merge_and_unload()  # 合并后返回的是原始模型
    model = model.half()  # 再次半精度转换以防止类型冲突

print("✅ 所有 LoRA 合并完成！")

# ======= 保存合并后的模型权重 =======
save_path = os.path.abspath(save_path)
os.makedirs(save_path, exist_ok=True)
print(f"💾 正在保存到：{save_path}")
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print("🎉 模型保存完成！你现在可以像普通 TinyChart 模型一样加载它了。")