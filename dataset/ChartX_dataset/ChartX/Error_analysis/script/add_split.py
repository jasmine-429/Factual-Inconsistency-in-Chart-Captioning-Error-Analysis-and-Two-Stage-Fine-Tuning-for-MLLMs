import json
from collections import OrderedDict

# 输入和输出文件路径
input_path = "/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/all_type/all_typr.jsonl"
output_path = "/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/all_type/all_with_split.jsonl"

# 模型类别映射字典
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

# 处理文件：读取、插入字段、写入
with open(input_path, "r", encoding="utf-8") as infile, \
     open(output_path, "w", encoding="utf-8") as outfile:
    for line in infile:
        item = json.loads(line)
        model_name = item.get("model_name", "")
        split_value = split_mapping.get(model_name, "Unknown")

        # 构造有序字典并插入split字段在model_name后
        new_item = OrderedDict()
        for key, value in item.items():
            new_item[key] = value
            if key == "model_name":
                new_item["split"] = split_value

        outfile.write(json.dumps(new_item, ensure_ascii=False) + "\n")