import os
import json
import traceback
from PIL import Image
from tqdm import tqdm
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from peft import PeftModel

# ========= 配置 =========
torch.set_grad_enabled(False)

model_path = "/data/jguo376/pretrained_models/Qwen2.5-VL-7B-Instruct"
lora_path = "/data/jguo376/project/llama_factory/saves/Qwen2-VL-7B-Instruct/lora/dpo_sft/checkpoint-1062"
use_lora = True

input_json = "/data/jguo376/project/dataset/test_dataset/ChartX/test_eva_data/data/eva_test.json"
output_json = "/data/jguo376/project/model/Qwen_VL_chat/true_dpo/chartx_caption_dpo_sft.json"
base_dir = "/data/jguo376/project/dataset/ChartX_dataset/ChartX"
max_items = None  # 设置为 None 表示处理全部

# ========= 加载模型 =========
base_model = AutoModelForVision2Seq.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
).eval()

if use_lora:
    model = PeftModel.from_pretrained(base_model, lora_path).eval()
else:
    model = base_model

print(f"🔍 模型类型: {type(model)}")
if hasattr(model, "base_model") and "peft" in str(type(model)).lower():
    print("✅ 成功加载 LoRA")
else:
    print("⚠️ 未成功加载 LoRA")

# ========= 加载 processor =========
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

# ========= 加载输入数据 =========
with open(input_json, "r", encoding="utf-8") as f:
    data_list = json.load(f)

os.makedirs(os.path.dirname(output_json), exist_ok=True)

# ========= 加载断点（已处理的样本）=========
if os.path.exists(output_json):
    with open(output_json, "r", encoding="utf-8") as f:
        try:
            existing_results = json.load(f)
        except json.JSONDecodeError:
            existing_results = []
    existing_imgs = set(item["img"] for item in existing_results)
else:
    existing_results = []
    existing_imgs = set()

results = existing_results.copy()

# ========= 推理主循环 =========
for idx, item in enumerate(tqdm(data_list, desc="Generating captions")):
    if max_items is not None and idx >= max_items:
        break

    if item["img"] in existing_imgs:
        print(f"⏩ 跳过已处理图像: {item['img']}")
        continue

    try:
        image_path = os.path.join(base_dir, item["img"].lstrip("./"))
        if not os.path.exists(image_path):
            print(f"⚠️ 图像不存在: {image_path}")
            continue

        image = Image.open(image_path).convert("RGB")

        # ✅ 构造与训练完全一致的 prompt
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Please describe the chart."}
                ]
            }
        ]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=[prompt], images=[image], return_tensors="pt").to(model.device)

        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id
        )

        caption_raw = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        caption = caption_raw.split("assistant\n", 1)[-1].strip()

        # ✅ 打印部分输出做调试
        print(f"📝 生成内容: {caption[:80]}")

        item["model_name"] = model_path
        item["generated_caption"] = caption
        results.append(item)
        existing_imgs.add(item["img"])

        # ✅ 保存中间结果（断点续跑）
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    except Exception as e:
        print(f"❌ 错误处理样本 {idx}: {e}")
        traceback.print_exc()

print(f"🎉 推理完成，输出保存在：{output_json}")

