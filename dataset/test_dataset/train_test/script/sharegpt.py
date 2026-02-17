import json

input_path = "/data/jguo376/project/dataset/test_dataset/train_test/dataset/total/sample__sft.json"
output_path = "/data/jguo376/project/dataset/test_dataset/train_test/dataset/total/chart_entail_sharegpt.json"

with open(input_path, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

converted_data = []
for item in raw_data:
    new_item = {
        "messages": [],
        "images": [item["image"]]
    }
    for conv in item["conversations"]:
        new_item["messages"].append({
            "role": "user" if conv["from"] == "human" else "assistant",
            "content": conv["value"] if conv["from"] == "human" else conv["value"]
        })
    converted_data.append(new_item)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(converted_data, f, indent=2, ensure_ascii=False)

print(f"✅ 转换完成，输出文件：{output_path}")
