import json
import random
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path

# ===== 输入输出路径 =====
input_path = "/data/jguo376/project/dataset/test_dataset/ChartX/test_data/dataset/test_samples_id.json"  # 原始图表数据路径
output_path = "/data/jguo376/project/dataset/test_dataset/ChartX/test_data/dataset/error_data/ooc_error_augmented.json"
log_path = "/data/jguo376/project/dataset/test_dataset/ChartX/test_data/dataset/error_data/error_log/ooc_error_log.txt"

# ===== 控制错配比例（0.0~1.0）=====
same_ratio = 0.6  # 同类型错配占比，剩下的为跨类型错配

# ===== 加载数据 =====
with open(input_path, "r") as f:
    data = json.load(f)

# ===== 按 chart_type 分组 =====
type2items = defaultdict(list)
for item in data:
    type2items[item["chart_type"]].append(item)

augmented = []
logs = []

# ===== 注入 ooc 错误 =====
for item in tqdm(data, desc="Injecting ooc_error"):
    base_type = item["chart_type"]
    source_img = item["img"]
    source_imgname = item["imgname"]

    r = random.random()

    # ===== 方法 1：同类型错配 =====
    if r < same_ratio:
        same_type_pool = [i for i in type2items[base_type] if i["img"] != source_img]
        if same_type_pool:
            swap_item = random.choice(same_type_pool)
            new_item = {
                "chart_type": base_type,
                "img": item["img"],
                "imgname": item["imgname"],
                "id": item["id"] + "_ooc",
                "source": item["source"],
                "sentence": swap_item["sentence"],
                "label": 0,
                "error": "ooc_error",
                "method": "same_type_swap"
            }
            augmented.append(new_item)
            logs.append(f"[SameType] {source_imgname} ← caption from {swap_item['imgname']}")
        continue

    # ===== 方法 2：跨类型错配 =====
    other_types = [t for t in type2items if t != base_type]
    if other_types:
        rand_other_type = random.choice(other_types)
        other_pool = type2items[rand_other_type]
        if other_pool:
            swap_item = random.choice(other_pool)
            new_item = {
                "chart_type": base_type,
                "img": item["img"],
                "imgname": item["imgname"],
                "id": item["id"] + "_ooc",
                "source": item["source"],
                "sentence": swap_item["sentence"],
                "label": 0,
                "error": "ooc_error",
                "method": "cross_type_swap"
            }
            augmented.append(new_item)
            logs.append(f"[CrossType] {source_imgname} ← caption from {swap_item['imgname']} ({rand_other_type})")

# ===== 保存输出文件 =====
Path(output_path).parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w") as f:
    json.dump(augmented, f, indent=2)

with open(log_path, "w") as f:
    f.write("\n".join(logs))

print(f"✅ ooc_error 注入完成，共生成 {len(augmented)} 条")
print(f"📄 日志写入：{log_path}")
