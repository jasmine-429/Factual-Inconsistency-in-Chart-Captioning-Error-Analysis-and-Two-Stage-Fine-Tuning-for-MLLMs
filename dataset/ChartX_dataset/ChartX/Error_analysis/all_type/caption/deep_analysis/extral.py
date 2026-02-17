import json
import os

# 输入输出路径
input_path = "/data/jguo376/project/dataset/ChartX_dataset/ChartX/ChartX_annotation.json"
output_path = "/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/all_type/caption/deep_analysis/caption_all/chartx_selected_fields.json"

# 加载原始数据
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 提取指定字段
selected_data = []
for entry in data:
    selected_entry = {
        "chart_type": entry.get("chart_type"),
        "imgname": entry.get("imgname"),
        "img": entry.get("img"),
        "topic": entry.get("topic"),
        "title": entry.get("title"),
        "csv": entry.get("csv")
    }
    selected_data.append(selected_entry)

# 保存到新文件
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(selected_data, f, indent=2, ensure_ascii=False)

print(f"✅ 已保存提取字段到: {output_path}")
