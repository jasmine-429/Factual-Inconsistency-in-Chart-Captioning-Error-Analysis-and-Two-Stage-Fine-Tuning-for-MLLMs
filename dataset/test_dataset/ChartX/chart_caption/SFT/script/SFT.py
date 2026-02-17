import json
import os

# 读取原始数据
with open("/Users/gjx/Desktop/test_dataset/ChartX/train_samples_with_QA.json", "r") as f:
    raw_data = json.load(f)

output_data = []

for item in raw_data:
    if "description" not in item or not item["description"].strip():
        continue  # 跳过无 description 的项

    image_path = item["img"]
    description = item["description"].strip()

    new_item = {
        "messages": [
            {
                "content": "<image>Please describe the chart.",
                "role": "user"
            },
            {
                "content": description,
                "role": "assistant"
            }
        ],
        "images": [
            image_path
        ]
    }

    output_data.append(new_item)

# 保存新数据格式
with open("/Users/gjx/Desktop/test_dataset/ChartX/Chart_caption/SFT/data/chart_caption_sft.json", "w") as f:
    json.dump(output_data, f, indent=2)
