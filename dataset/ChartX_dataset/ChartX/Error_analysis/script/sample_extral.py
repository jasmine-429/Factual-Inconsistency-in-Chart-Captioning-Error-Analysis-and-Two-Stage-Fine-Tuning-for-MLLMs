# 从 ChartX_annotation.json 中读取，并对 bar、line、pie 三类图表类型抽样
import json
import random

# 载入 annotation 文件
annotation_file = "/data/jguo376/project/dataset/ChartX_dataset/ChartX/ChartX_annotation.json"
with open(annotation_file, "r") as f:
    annotations = json.load(f)

# 图表类型归类
bar_types = {"bar_chart", "bar_chart_num"}
line_types = {"line_chart", "line_chart_num"}
pie_types = {"pie_chart"}

bar_samples = []
line_samples = []
pie_samples = []

for item in annotations:
    chart_type = item.get("chart_type")
    if chart_type in bar_types:
        bar_samples.append(item)
    elif chart_type in line_types:
        line_samples.append(item)
    elif chart_type in pie_types:
        pie_samples.append(item)

# 抽样
sampled = []
sampled.extend(random.sample(bar_samples, min(25, len(bar_samples))))
sampled.extend(random.sample(line_samples, min(25, len(line_samples))))
sampled.extend(random.sample(pie_samples, min(25, len(pie_samples))))

# 输出为 JSONL 文件
output_path = "/data/jguo376/project/dataset/ChartX_dataset/ChartX/ChartX_sampled.jsonl"
with open(output_path, "w") as fout:
    for entry in sampled:
        # 仅保留指定字段
        trimmed = {
            "chart_type": entry.get("chart_type"),
            "imgname": entry.get("imgname"),
            "img": entry.get("img"),
            "topic": entry.get("topic"),
            "title": entry.get("title"),
            "csv": entry.get("csv")
        }
        fout.write(json.dumps(trimmed) + "\n")

output_path
