import json

with open("/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/all_type/caption/deep_analysis/caption_all/llava_caption_output.json", "r") as f:
    data = json.load(f)

print("Total samples:", len(data))
