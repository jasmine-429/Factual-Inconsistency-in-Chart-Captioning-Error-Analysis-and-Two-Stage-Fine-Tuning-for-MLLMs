import json
import re
import pandas as pd
import os
from io import StringIO

# === 配置输入 JSON 文件路径列表 ===
input_json_list = [
    "/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/all_type/caption/deep_analysis/caption_all/chartvlm_caption_output.json",
    "/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/all_type/caption/deep_analysis/caption_all/internlm_caption_output_org.json",
    "/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/all_type/caption/deep_analysis/caption_all/llava_caption_output.json",
    "/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/all_type/caption/deep_analysis/caption_all/matcha_caption_output.json",
    "/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/all_type/caption/deep_analysis/caption_all/mmca_caption_output.json",
    "/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/all_type/caption/deep_analysis/caption_all/pix2struct_caption_output.json",
    "/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/all_type/caption/deep_analysis/caption_all/qwen_caption_output.json",
    "/data/jguo376/project/model/chartInstruction/chartx_caption.json",
    "/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/all_type/caption/deep_analysis/caption_all/unichart_caption_output.json"
]

# === 更改输出目录，避免权限问题 ===
output_dir = "/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/all_type/caption/Recall"
os.makedirs(output_dir, exist_ok=True)

def extract_numbers(text):
    if not text:
        return []
    # 匹配：整数、小数、百分比、千分位
    pattern = r'-?\d+(?:,\d{3})*(?:\.\d+)?%?'
    matches = re.findall(pattern, text)
    numbers = []
    for m in matches:
        m = m.replace(",", "")  # 去掉千分位
        if m.endswith('%'):
            m = m[:-1]  # 去掉百分号
        try:
            numbers.append(float(m))
        except:
            continue
    return numbers

def extract_csv_numbers_from_string(csv_str):
    if not csv_str.strip():
        return []
    # 转义修复：有些数据里 \t 和 \n 是字符串字面量
    csv_str = csv_str.encode().decode('unicode_escape')
    try:
        df = pd.read_csv(StringIO(csv_str), sep="\t")
    except Exception as e:
        print(f"[!] CSV解析失败: {e}，原始CSV: {csv_str[:100]}")
        return []
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    numbers = []
    for col in numeric_cols:
        numbers += df[col].dropna().astype(float).tolist()
    return numbers

def calculate_number_recall(caption_numbers, csv_numbers, eps=0.1):
    if len(csv_numbers) == 0 or len(caption_numbers) == 0:
        return 0.0
    matched = 0
    unmatched_csv = csv_numbers.copy()
    for csv_num in csv_numbers:
        for i, cap_num in enumerate(caption_numbers):
            if abs(cap_num - csv_num) <= eps:
                matched += 1
                caption_numbers.pop(i)
                break
    return matched / len(csv_numbers)

all_results = []

# === 批量处理所有模型 JSON ===
for json_path in input_json_list:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for idx, item in enumerate(data):
        chart_type = item.get("chart_type", "unknown")
        model_name = item.get("model_name", "unknown")
        csv_str = item.get("csv", "").strip()
        caption = item.get("generated_caption", "").strip()
        imgname = item.get("imgname", f"{model_name}_{idx}")

        csv_numbers = extract_csv_numbers_from_string(csv_str)
        caption_numbers = extract_numbers(caption)
        recall = calculate_number_recall(caption_numbers.copy(), csv_numbers)

        all_results.append({
            "imgname": imgname,
            "model": model_name,
            "chart_type": chart_type,
            "csv_numbers_count": len(csv_numbers),
            "caption_numbers_count": len(caption_numbers),
            "matched_count": int(recall * len(csv_numbers)) if len(csv_numbers) > 0 else 0,
            "number_recall": round(recall, 4)
        })

# === 输出为DataFrame并汇总 ===
df = pd.DataFrame(all_results)

# 按模型汇总
by_model = df.groupby("model").agg({
    "number_recall": ["mean", "count"],
    "csv_numbers_count": "sum",
    "matched_count": "sum"
}).reset_index()
by_model.columns = ["model", "avg_recall", "sample_count", "total_csv_numbers", "total_matched"]
by_model["avg_recall"] = round(by_model["avg_recall"] * 100, 2)

# 按模型+图表类型汇总
by_model_chart = df.groupby(["model", "chart_type"]).agg({
    "number_recall": ["mean", "count"],
    "csv_numbers_count": "sum",
    "matched_count": "sum"
}).reset_index()
by_model_chart.columns = ["model", "chart_type", "avg_recall", "sample_count", "total_csv_numbers", "total_matched"]
by_model_chart["avg_recall"] = round(by_model_chart["avg_recall"] * 100, 2)

# === 保存输出 ===
df.to_csv(os.path.join(output_dir, "value_recall_sample_level_in.csv"), index=False)
by_model.to_csv(os.path.join(output_dir, "value_recall_by_model_in.csv"), index=False)
by_model_chart.to_csv(os.path.join(output_dir, "value_recall_by_model_chart_in.csv"), index=False)

# === 打印核心摘要 ===
print("=== 按模型的 Value Recall（%） ===")
print(by_model[["model", "avg_recall", "sample_count", "total_matched", "total_csv_numbers"]].to_string(index=False))

print("\n=== 样本示例（前5条） ===")
print(df[["imgname", "model", "number_recall", "csv_numbers_count", "caption_numbers_count"]].head(5).to_string(index=False))
