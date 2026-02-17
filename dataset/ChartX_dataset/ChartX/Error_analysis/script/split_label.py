import json

# 定义模型名称到类别（split）字段的映射
model_to_split = {
    "InternLM-XC-v2-7B": "General_MLLM",
    "Qwen-VL-9.6B": "General_MLLM",
    "LLaVA-v1.5": "General_MLLM",
    "UniChart-201M": "Specialist_Chart_Model",
    "Matcha-282M": "Specialist_Chart_Model",
    "Pix2Struct-282M": "Specialist_Chart_Model",
    "TinyChart-3B": "Chart_MLLM",
    "MMCA-7B": "Chart_MLLM",
    "ChartVLM-13B": "Chart_MLLM"
}

# 文件路径
input_path = "/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/merged_output.jsonl"
output_path = "/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/merged_output_with_split.jsonl"

with open(input_path, "r") as infile, open(output_path, "w") as outfile:
    for line in infile:
        data = json.loads(line)
        model_name = data.get("model_name")
        split_value = model_to_split.get(model_name, "Unknown")
        
        # 重新组织字段顺序，使 split 紧跟在 model_name 后
        new_data = {}
        for key in list(data.keys()):
            new_data[key] = data[key]
            if key == "model_name":
                new_data["split"] = split_value

        outfile.write(json.dumps(new_data, ensure_ascii=False) + "\n")

print(f"✅ 写入完成：{output_path}")
