import json
import os
#2 倒3-ring前四 倒1 各缺4个

# 所有待合并的 JSON 文件路径（按需修改）
json_paths = [
    "/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/all_type/caption/deep_analysis/caption_all/chartvlm_caption_output.json",
    "/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/all_type/caption/deep_analysis/caption_all/internlm_caption_output_org.json",
    "/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/all_type/caption/deep_analysis/caption_all/llava_caption_output.json",
    "/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/all_type/caption/deep_analysis/caption_all/matcha_caption_output.json",
    "/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/all_type/caption/deep_analysis/caption_all/mmca_caption_output.json",
    "/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/all_type/caption/deep_analysis/caption_all/pix2struct_caption_output.json",
    "/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/all_type/caption/deep_analysis/caption_all/qwen_caption_output.json",
    "/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/all_type/caption/deep_analysis/caption_all/tinychart_caption_output.json",
    "/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/all_type/caption/deep_analysis/caption_all/unichart_caption_output.json"
    
]

merged_data = []

# 逐个加载并合并
for path in json_paths:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        if isinstance(data, list):
            merged_data.extend(data)
        else:
            merged_data.append(data)  # 如果是单条 dict 也加入

# 保存到一个新 JSON 文件
output_path = "/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/all_type/caption/deep_analysis/caption_all/merged_output.json"
with open(output_path, "w", encoding="utf-8") as f_out:
    json.dump(merged_data, f_out, indent=2, ensure_ascii=False)

print(f"✅ 合并完成，共 {len(merged_data)} 条记录，保存至 {output_path}")
