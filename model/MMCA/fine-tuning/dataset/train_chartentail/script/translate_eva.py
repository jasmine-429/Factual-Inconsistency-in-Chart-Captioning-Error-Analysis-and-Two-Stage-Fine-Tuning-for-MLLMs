import json
import random
from pathlib import Path
from tqdm import tqdm

#生成训练的验证集

# === 配置路径 ===
input_path = "/data/jguo376/project/dataset/test_dataset/ChartX/test_data/dataset/test_sft.json"
output_path = "/data/jguo376/project/model/MMCA/fine-tuning/dataset/train_chartentail/data/mmca_eva_200.json"
n_samples = 200

# === 文件名转换函数 ===
def derive_new_name(rel_path: str) -> str:
    p = Path(rel_path)
    parts = p.parts
    chart_type = parts[0] if len(parts) >= 2 else ""
    base_name = Path(parts[-1]).stem
    ext = Path(parts[-1]).suffix
    if base_name.isdigit():
        return f"{chart_type}_{base_name}{ext}"
    else:
        return base_name + ext

# === 读取原始数据 ===
with open(input_path, "r") as f:
    raw_data = json.load(f)

# === 转换格式 ===
converted_data = []
for item in tqdm(raw_data):
    image_path = item["image"].lstrip("./")
    image_name = derive_new_name(image_path)
    user_query = item["conversations"][0]["value"]
    assistant_answer = item["conversations"][1]["value"]

    sharegpt_text = (
        "The following is a conversation between a curious human and AI assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
        f"Human: <image>\n{user_query}\nAI: {assistant_answer}"
    )

    converted_data.append({
        "image": [image_name],
        "text": sharegpt_text,
        "task_type": "sharegpt_chat_sft"
    })

# === 随机抽样 200 条 ===
random.seed(42)
sampled = random.sample(converted_data, min(n_samples, len(converted_data)))

# === 写入为 JSONL，每行为一个 JSON 对象 ===
with open(output_path, "w") as f:
    for item in sampled:
        json.dump(item, f, ensure_ascii=False)
        f.write("\n")

print(f"✅ 已保存 JSONL 到 {output_path}，共 {len(sampled)} 条样本")
