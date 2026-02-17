import json

input_json_path = "/data/jguo376/project/dataset/test_dataset/ChartX/test_data/dataset/combined_chocolate_dataset.json"
output_jsonl_path = "/data/jguo376/project/dataset/test_dataset/ChartX/test_data/dataset/combined_dataset.jsonl"

with open(input_json_path, "r") as f:
    data = json.load(f)

with open(output_jsonl_path, "w") as fout:
    for item in data:
        fout.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"✅ Converted {len(data)} items to JSONL format at: {output_jsonl_path}")
