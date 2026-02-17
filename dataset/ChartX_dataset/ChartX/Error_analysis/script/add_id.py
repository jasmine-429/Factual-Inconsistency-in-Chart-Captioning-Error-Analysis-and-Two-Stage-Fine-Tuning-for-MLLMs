import json
from collections import OrderedDict

# 文件路径
input_path = "/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/all_type/all_with_split.jsonl"
output_path = "/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/all_type/all_with_split_id.jsonl"

# 模型分类映射
split_mapping = {
    "InternLM-XC-v2-7B": "General_MLLM",
    "Qwen-VL-9.6B": "General_MLLM",
    "LLaVA-v1.5-13B": "General_MLLM",
    "UniChart-201M": "Specialist_Chart_Model",
    "Matcha-282M": "Specialist_Chart_Model",
    "Pix2Struct-282M": "Specialist_Chart_Model",
    "TinyChart-3B": "Chart_MLLM",
    "MMCA-7B": "Chart_MLLM",
    "ChartVLM-13B": "Chart_MLLM"
}

def is_pure_number(s):
    return s.isdigit()

# 主处理流程
with open(input_path, "r", encoding="utf-8") as infile, \
     open(output_path, "w", encoding="utf-8") as outfile:
    for line in infile:
        item = json.loads(line)
        model_name = item.get("model_name", "")
        chart_type = item.get("chart_type", "")
        imgname = item.get("imgname", "")

        # 处理 imgname（如有必要前加 chart_type）
        if is_pure_number(imgname) and chart_type:
            imgname = f"{chart_type}_{imgname}"

        # 构造 _id 字段
        item_id = f"{model_name}_{imgname}"

        # 构造有序字段，确保顺序：model_name → split → _id → 其他
        new_item = OrderedDict()
        for key, value in item.items():
            new_item[key] = value
            if key == "model_name":
                new_item["split"] = split_mapping.get(model_name, "Unknown")
                new_item["_id"] = item_id

        outfile.write(json.dumps(new_item, ensure_ascii=False) + "\n")
