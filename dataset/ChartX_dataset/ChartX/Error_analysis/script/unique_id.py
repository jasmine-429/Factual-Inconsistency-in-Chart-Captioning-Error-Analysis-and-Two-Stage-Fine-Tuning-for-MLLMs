import json

input_path = "/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/all_type/all_with_split_id.jsonl"
seen_ids = set()
duplicates = set()

with open(input_path, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        _id = item.get("_id", None)
        if _id in seen_ids:
            duplicates.add(_id)
        else:
            seen_ids.add(_id)

if duplicates:
    print(f"❌ Found {len(duplicates)} duplicate _id(s):")
    for dup in sorted(duplicates):
        print(dup)
else:
    print("✅ All _id values are unique.")
