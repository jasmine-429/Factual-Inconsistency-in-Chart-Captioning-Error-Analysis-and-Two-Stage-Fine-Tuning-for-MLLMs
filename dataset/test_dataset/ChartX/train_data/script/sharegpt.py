import json

input_path = "/data/jguo376/project/dataset/test_dataset/ChartX/test_data/dataset/test_sft.json"
output_path = "/data/jguo376/project/dataset/test_dataset/ChartX/test_data/dataset/chart_entail_test.json"

with open(input_path, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

converted_data = []
for item in raw_data:
    # 替换图像路径：去掉开头 ./ ，加上 Chart/
    image_path = "Chart/" + item["image"].lstrip("./")
    new_item = {
        "messages": [],
        "images": [image_path]
    }
    for conv in item["conversations"]:
        role = "user" if conv["from"] == "human" else "assistant"
        content = "<image>" + conv["value"] if role == "user" else conv["value"]
        new_item["messages"].append({
            "role": role,
            "content": content
        })
    converted_data.append(new_item)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(converted_data, f, indent=2, ensure_ascii=False)

print(f"✅ 转换完成，输出文件：{output_path}")
