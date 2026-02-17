import json

# 替换为你的实际 JSON 文件路径
input_json_path = "/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/all_type/caption/deep_analysis/caption_all/matcha_caption_output.json"


with open(input_json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

count = 0

for item in data:
    caption = item.get("generated_caption", "").lower()
    title = item.get("title", "").lower()

    triggered = False
    if "in 2019" in caption and "in 2019" not in title:
        triggered = True
    if "in the united states" in caption and "in the united states" not in title:
        triggered = True

    if triggered:
        count += 1

print(f"符合条件的条目数量: {count}")
