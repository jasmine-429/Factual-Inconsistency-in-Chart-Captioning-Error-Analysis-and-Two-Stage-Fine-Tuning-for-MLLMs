import json
from collections import defaultdict

# 读取原始 JSON 数据
with open("/Users/gjx/Desktop/test_dataset/ChartX/Chart_caption/data/augmented_data.json", "r") as f:
    data = json.load(f)

# 按 (imgname, source) 分组，分别收集 consistent / inconsistent caption
grouped = defaultdict(lambda: {"true": [], "false": []})

for item in data:
    key = (item["imgname"], item["source"])
    if item["caption_consistent"]:
        grouped[key]["true"].append(item)
    else:
        grouped[key]["false"].append(item)

# 构建 DPO 格式数据（图 + 正确 caption + 错误 caption）
dpo_data = []

for (imgname, source), group in grouped.items():
    true_list = group["true"]
    false_list = group["false"]

    if not true_list or not false_list:
        continue  # 需要正负样本才构造

    image_path = true_list[0]["img"] if true_list else false_list[0]["img"]

    # 每个 true caption 对应一个 false caption 构造 pair（如需限制数量可加 random.sample）
    for true_item in true_list:
        for false_item in false_list:
            dpo_data.append({
                "conversations": [
                    {
                        "from": "human",
                        "value": "Please describe the chart."
                    }
                ],
                "image": image_path,
                "chosen": {
                    "from": "gpt",
                    "value": true_item["caption"]
                },
                "rejected": {
                    "from": "gpt",
                    "value": false_item["caption"]
                }
            })

# 保存输出
with open("/Users/gjx/Desktop/test_dataset/ChartX/Chart_caption/data/chart_caption_dpo.json", "w") as f:
    json.dump(dpo_data, f, indent=2)
