#提取测试集
import json
import os
import random
from collections import defaultdict

# ========== 文件路径配置 ==========
annotation_path = "/data/jguo376/project/dataset/ChartX_dataset/ChartX/ChartX_annotation.json"
evaluate_path = "/data/jguo376/project/dataset/ChartX_dataset/ChartX/chart_evaluate.jsonl"
output_path = "//data/jguo376/project/dataset/test_dataset/ChartX/test_samples.json"

# ========== 图表类型与抽样数量配置 ==========
type_to_num = {
    "bar_chart": 50,
    "line_chart": 50,
    "pie_chart": 50,
    "bar_chart_num": 50,
    "line_chart_num": 50,
    "rings": 30,
    "rose": 30,
    "area_chart": 30,
    "heatmap": 30,
    "3D-Bar": 30,
    "box": 30,
    "bubble": 30,
    "candlestick": 30,
    "funnel": 30,
    "histogram": 30,
    "multi-axes": 30,
    "radar": 30,
    "treemap": 30,
}

# ========== 读取已使用图像列表 ==========
used_imgs = set()
with open(evaluate_path, "r") as f:
    for line in f:
        try:
            item = json.loads(line)
            used_imgs.add(item.get("img"))
        except json.JSONDecodeError:
            continue

# ========== 加载 ChartX 注释数据 ==========
with open(annotation_path, "r") as f:
    all_data = json.load(f)

# ========== 分类收集图表样本（排除已使用图像） ==========
type_to_items = defaultdict(list)
for item in all_data:
    chart_type = item.get("chart_type")
    img_path = item.get("img")
    if chart_type in type_to_num and img_path not in used_imgs:
        type_to_items[chart_type].append(item)

# ========== 每类图表按需抽样 ==========
final_samples = []
for chart_type, required_num in type_to_num.items():
    candidates = type_to_items.get(chart_type, [])
    if len(candidates) >= required_num:
        sampled = random.sample(candidates, required_num)
    else:
        print(f"⚠️ 类型 {chart_type} 仅有 {len(candidates)} 个候选样本，不足 {required_num}")
        sampled = candidates
    final_samples.extend(sampled)

# ========== 保存为 JSON 文件 ==========
with open(output_path, "w") as fout:
    json.dump(final_samples, fout, indent=2)

print(f"✅ 筛选完成，输出样本数量: {len(final_samples)}，保存路径: {output_path}")
