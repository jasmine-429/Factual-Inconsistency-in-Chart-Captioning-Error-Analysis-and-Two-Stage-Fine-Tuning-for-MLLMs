import json

# 输入和输出路径
input_json = "/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/sampled_exclude_5/sentence_dataset/exclude_5_merged_output.json"
output_jsonl = "/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/sampled_exclude_5/sentence_dataset/exclude_5_merged_output.jsonl"

# 加载 JSON 列表
with open(input_json, "r", encoding="utf-8") as f:
    data = json.load(f)

# 写入 JSONL，每行一个 JSON 对象
with open(output_jsonl, "w", encoding="utf-8") as f:
    for item in data:
        json_line = json.dumps(item, ensure_ascii=False)
        f.write(json_line + "\n")

print(f"✅ 转换完成：{output_jsonl} 共 {len(data)} 行")
