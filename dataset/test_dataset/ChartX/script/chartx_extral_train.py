import json

# ===== 输入输出路径 =====
input_path = "/data/jguo376/project/dataset/test_dataset/ChartX/test_data/test_samples.json"
output_path = "/data/jguo376/project/dataset/test_dataset/ChartX/test_data/dataset/train_samples_extracted.json"

# ===== 加载数据 =====
with open(input_path, "r") as f:
    data = json.load(f)

# ===== 提取需要字段 =====
filtered = []
for item in data:
    new_item = {
        "chart_type": item.get("chart_type"),
        "img": item.get("img"),
        "imgname": item.get("imgname"),
        "csv": item.get("csv"),
        "title": item.get("title"),
        "topic": item.get("topic"),
        "description": item.get("description", {}).get("output", ""),
        "summarization": item.get("summarization", {}).get("output") or item.get("summarization", {}).get("ouput_put", "")
    }
    filtered.append(new_item)

# ===== 保存结果 =====
with open(output_path, "w") as f:
    json.dump(filtered, f, indent=2)

print(f"✅ 提取完成，共 {len(filtered)} 条，保存至: {output_path}")
