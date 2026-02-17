import os
import json

# 路径配置
dataset_root = "/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/all_type/caption/deep_analysis/caption_all"
input_json = os.path.join(dataset_root, "chartx_selected_fields.json")
output_json = os.path.join(dataset_root, "chartvlm_caption_output.json")
chart_root = "/data/jguo376/project/dataset/ChartX_dataset/ChartX"
missing_json = os.path.join(dataset_root, "chartve_missing_subset.json")  # ✅ 保存未处理样本

# 加载原始数据
with open(input_json, "r", encoding="utf-8") as f:
    data_list = json.load(f)

# 加载已完成数据
processed_imgs = set()
if os.path.exists(output_json):
    with open(output_json, "r", encoding="utf-8") as f:
        output_data = json.load(f)
        for entry in output_data:
            processed_imgs.add(entry["img"])

# 生成未处理样本
missing_items = []
for item in data_list:
    rel_path = item["img"]
    abs_path = os.path.join(chart_root, rel_path.lstrip("./"))
    if abs_path not in processed_imgs:
        missing_items.append(item)

# 保存未完成样本
with open(missing_json, "w", encoding="utf-8") as f:
    json.dump(missing_items, f, indent=2, ensure_ascii=False)

print(f"✅ 共找到未处理样本 {len(missing_items)} 条，已保存至：{missing_json}")
