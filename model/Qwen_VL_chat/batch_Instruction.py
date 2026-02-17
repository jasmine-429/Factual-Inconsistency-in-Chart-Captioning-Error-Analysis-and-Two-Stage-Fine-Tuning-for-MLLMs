import os
import json
from PIL import Image
from tqdm import tqdm
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq


os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# ========= 模型 & 路径配置 =========
model_id = "/data/jguo376/pretrained_models/Qwen2.5-VL-7B-Instruct"
base_dir = "/data/jguo376/project/dataset/ChartX_dataset/ChartX"
input_json = "/data/jguo376/project/dataset/ChartX_dataset/chartx.json"
output_json = "/data/jguo376/project/dataset/test_dataset/train_test/dataset/qwen_caption_output_instru.json"
max_items = 5  # 设置为整数可限制推理样本数

# ========= 加载模型 =========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForVision2Seq.from_pretrained(  # ✅ 使用 AutoModelForVision2Seq
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True  # ✅ 必须加！
).eval()
processor = AutoProcessor.from_pretrained(model_id)

# ========= 加载数据 =========
with open(input_json, "r", encoding="utf-8") as f:
    data_list = json.load(f)

results = []
for idx, item in enumerate(tqdm(data_list, desc="Generating captions")):
    if max_items is not None and idx >= max_items:
        break
    try:
        image_path = os.path.join(base_dir, item["img"].lstrip("./"))
        if not os.path.exists(image_path):
            print(f"⚠️ Image not found: {image_path}")
            continue
        image = Image.open(image_path).convert("RGB")

        # 构建 messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Provide a short analytical description of the chart, including specific values, comparisons, and trends."}
                ]
            }
        ]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=[prompt], images=[image], return_tensors="pt").to(model.device)
        
        # 推理
        output_ids = model.generate(**inputs, max_new_tokens=150)
        caption_raw = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        caption = caption_raw.split("assistant\n", 1)[-1].strip()  # ✅ 去除前缀

        item["model_name"] = model_id
        item["generated_caption"] = caption
        results.append(item)

        print(f"✅ {item.get('imgname', f'idx_{idx}')} => {caption[:60]}...")

    except Exception as e:
        print(f"❌ Failed to process line {idx}: {e}")

# ========= 写出结果 =========
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"🎉 批处理完成，结果保存在：{output_json}")
