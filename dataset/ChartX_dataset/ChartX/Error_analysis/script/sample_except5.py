import json
import random
from collections import defaultdict

# 要排除的五类图表类型
excluded_types = {"bar_chart", "bar_chart_num", "line_chart", "line_chart_num", "pie_chart"}

# 加载数据
input_path = "/data/jguo376/project/dataset/ChartX_dataset/ChartX/ChartX_annotation.json"
with open(input_path, "r") as f:
    data = json.load(f)

# 分类收集（排除指定类型）
type_to_items = defaultdict(list)
for item in data:
    chart_type = item.get("chart_type", "")
    if chart_type not in excluded_types:
        type_to_items[chart_type].append(item)

# 每类采样最多25条
sampled_data = []
for chart_type, items in type_to_items.items():
    sample_count = min(25, len(items))
    sampled_items = random.sample(items, sample_count)
    sampled_data.extend(sampled_items)
    print(f"{chart_type}: sampled {sample_count} / {len(items)}")

# 输出为 JSONL 文件
output_path = "/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/sampled_exclude_5/sampled_excluding_5types.jsonl"
with open(output_path, "w") as fout:
    for entry in sampled_data:
        trimmed = {
            "chart_type": entry.get("chart_type"),
            "imgname": entry.get("imgname"),
            "img": entry.get("img"),
            "topic": entry.get("topic"),
            "title": entry.get("title"),
            "csv": entry.get("csv")
        }
        fout.write(json.dumps(trimmed) + "\n")

print(f"✅ Saved sampled data to {output_path}")
