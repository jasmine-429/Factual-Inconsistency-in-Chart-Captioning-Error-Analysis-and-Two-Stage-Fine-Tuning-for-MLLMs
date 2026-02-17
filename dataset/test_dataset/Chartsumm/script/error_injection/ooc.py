import json
import random
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

# ===== 输入输出路径 =====
input_path = "/data/jguo376/project/dataset/test_dataset/Chartsumm/train_data/train_s/train_s_sentences.json"  # 原始数据路径
output_path = "/data/jguo376/project/dataset/test_dataset/Chartsumm/train_data/train_s/ooc_error_augmented.json"
log_path = "/data/jguo376/project/dataset/test_dataset/Chartsumm/train_data/train_s/error_log/ooc_error_log.txt"

# ===== 加载数据 =====
with open(input_path, "r") as f:
    data = json.load(f)

# ===== 将每张图按 img 分组 =====
img2samples = defaultdict(list)
for item in data:
    img2samples[item["img"]].append(item)

# ===== 为每张图选择一个 base sentence（如 summary 优先）=====
selected_sentences = []
for img, items in img2samples.items():
    # 优先选择 summary，没有就随机选一个
    summary_items = [i for i in items if i["source"] == "summary"]
    selected = summary_items[0] if summary_items else random.choice(items)
    selected_sentences.append(selected)

# ===== 生成 ooc 错配句子 =====
augmented = []
logs = []

for base_item in tqdm(selected_sentences, desc="Injecting ooc_error"):
    base_img = base_item["img"]
    base_id = base_item["id"]
    base_sentence = base_item["sentence"]

    # 从其他图中随机选一个不同的句子（错配）
    candidates = [i for i in selected_sentences if i["img"] != base_img]
    if not candidates:
        continue
    swap_item = random.choice(candidates)

    new_item = {
        "img": base_item["img"],
        "imgname": base_item["imgname"],
        "id": base_item["id"] + "_ooc",
        "source": base_item["source"],
        "sentence": swap_item["sentence"],
        "label": 0,
        "error": "ooc_error"
    }
    augmented.append(new_item)
    logs.append(f"[OOC] {base_item['imgname']} ← caption from {swap_item['imgname']}")

# ===== 保存输出 =====
Path(output_path).parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w") as f:
    json.dump(augmented, f, indent=2)

with open(log_path, "w") as f:
    f.write("\n".join(logs))

print(f"✅ ooc_error 注入完成，共生成 {len(augmented)} 条")
print(f"📄 日志写入：{log_path}")
