import sys
sys.path.append("/data/jguo376/project/model/mPLUG-Owl/mPLUG-Owl")

import os
import torch
from transformers import AutoTokenizer
from peft import PeftModel, LoraConfig, PeftType
from mplug_owl.modeling_mplug_owl import MplugOwlForConditionalGeneration

# === 路径设置 ===
base_model_path = "/data/jguo376/pretrained_models/mplug-owl-llama-7b"
mmca_adapter_path = "/data/jguo376/pretrained_models/MMCA_lora_adapter"
merged_model_output = "/data/jguo376/pretrained_models/mmca_merged_model"

# === 加载 base 模型 ===
print("Loading base model...")
base_model = MplugOwlForConditionalGeneration.from_pretrained(
    base_model_path,
    trust_remote_code=True,
    torch_dtype=torch.float16
)

# === 手动构造 LoRA config（防止 KeyError）===
print("Constructing fallback LoRA config...")
lora_config = LoraConfig.from_pretrained(mmca_adapter_path)
lora_config.peft_type = PeftType.LORA  # 明确指定类型

# === 加载 adapter ===
print("Loading MMCA adapter...")
model = PeftModel.from_pretrained(base_model, mmca_adapter_path, config=lora_config)

# === 合并 LoRA 到 base 模型 ===
print("Merging adapter into base model...")
model = model.merge_and_unload()

# === 保存合并模型 ===
print(f"Saving merged model to {merged_model_output}")
model.save_pretrained(merged_model_output)

# === 保存 tokenizer（推荐）===
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
tokenizer.save_pretrained(merged_model_output)

print("✅ Merge completed successfully.")
