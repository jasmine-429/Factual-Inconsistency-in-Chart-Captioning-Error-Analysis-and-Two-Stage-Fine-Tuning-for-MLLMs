import os
import json
from PIL import Image
from tqdm import tqdm
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

# ========= 路径配置 =========
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

model_id = "/data/jguo376/pretrained_models/Qwen2.5-VL-7B-Instruct"
base_dir = "/data/jguo376/project/dataset/ChartX_dataset/ChartX"
input_json = "/data/jguo376/project/dataset/test_dataset/ChartX/test_data/dataset/test_sft.json"
output_json = "/data/jguo376/project/dataset/test_dataset/ChartX/test_data/test_6600-all.json"
max_items = None  # 设为 None 表示不限制条数

# ========= 加载模型 =========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = AutoModelForVision2Seq.from_pretrained(
#    model_id,
#    torch_dtype=torch.float16,
#    device_map="auto",
#    trust_remote_code=True
#).eval()
from peft import PeftModel

base_model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

adapter_path = "/data/jguo376/project/llama_factory/src/saves/Qwen2-VL-7B-Instruct/lora/train_2025-07-29-23-35-38/checkpoint-1176"
model = PeftModel.from_pretrained(base_model, adapter_path).eval()
processor = AutoProcessor.from_pretrained(model_id)

# ========= 加载数据 =========
with open(input_json, "r", encoding="utf-8") as f:
    data_list = json.load(f)

# ========= 加载已有结果用于断点续跑 =========
if os.path.exists(output_json):
    with open(output_json, "r", encoding="utf-8") as f:
        existing_data = json.load(f)
        completed_ids = {item["id"] for item in existing_data}
        results = existing_data
else:
    completed_ids = set()
    results = []

# ========= 推理主循环 =========
for idx, item in enumerate(tqdm(data_list, desc="Entailment Inference")):
    if max_items is not None and idx >= max_items:
        break
    if item["id"] in completed_ids:
        continue

    try:
        # 加载图像
        image_path = os.path.join(base_dir, item["image"].lstrip("./"))
        if not os.path.exists(image_path):
            print(f"⚠️ Image not found: {image_path}")
            continue
        image = Image.open(image_path).convert("RGB")

        # 提取 prompt
        convs = item["conversations"]
        user_prompt = next((c["value"] for c in convs if c["from"] == "human"), None)
        if not user_prompt:
            print(f"⚠️ No user prompt found for {item['id']}")
            continue

        # 构造 messages & prompt
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_prompt}]}]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=[prompt], images=[image], return_tensors="pt").to(model.device)

        # 生成输出
        output_ids = model.generate(**inputs, max_new_tokens=128)
        prediction_raw = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

        # 提取 Assistant 的回答部分
        if "assistant\n" in prediction_raw:
            prediction = prediction_raw.split("assistant\n", 1)[-1].strip()
        else:
            prediction = prediction_raw.strip().split("\n")[-1].strip()

        # 保存至当前 item
        item["model_prediction"] = prediction
        item["model_answer_raw"] = prediction_raw
        item["question_type"] = "entailment"

        results.append(item)

        # ✅ 实时断点保存
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"[{idx}] ✅ {item['id']} | Pred: {prediction[:80]}")

    except Exception as e:
        print(f"❌ Failed on {item['id']}: {e}")

print(f"\n🎉 All finished! Results saved to: {output_json}")
