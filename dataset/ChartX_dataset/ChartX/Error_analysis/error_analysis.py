import json
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

# 输入文件路径
input_path = "/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/merged_output.jsonl"

# 读取并解析 jsonl 文件
records = []
with open(input_path, 'r') as f:
    for line in tqdm(f, desc="Parsing"):
        entry = json.loads(line)
        model = entry.get("model_name", "unknown_model")
        sentence_labels = entry.get("labels", [])
        for label_list in sentence_labels:
            for error_type in label_list:
                records.append({
                    "Model": model,
                    "Error Type": error_type
                })

# 转换为 DataFrame
df = pd.DataFrame(records)

# 统计每种错误类型的数量和比例
error_counts = df.groupby(["Model", "Error Type"]).size().reset_index(name="Count")
model_totals = df.groupby("Model").size().reset_index(name="Total")
result_df = pd.merge(error_counts, model_totals, on="Model")
result_df["Proportion"] = result_df["Count"] / result_df["Total"]

# 保存结果到 CSV（可选）
result_df.to_csv("/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/result/model_error_distribution.csv", index=False)

# 显示结果
print(result_df)


import matplotlib.pyplot as plt

# 原始标签 → 显示标签映射
label_mapping = {
    "value_error": "Value Error",
    "label_error": "Label Error",
    "trend_error": "Trend Error",
    "ooc_error": "Out Of Context Error",
    "magnitude_error": "Magnitude Error",
    "nonsense_error": "Nonsense Error"
}

# 定义颜色顺序（使用原始标签名）
error_order = list(label_mapping.keys())
color_map = {
    "value_error": "#1f77b4",
    "label_error": "#ff7f0e",
    "trend_error": "#d62728",
    "ooc_error": "#2ca02c",
    "magnitude_error": "#9467bd",
    "nonsense_error": "#c49c94"
}

# 创建透视表
pivot = result_df.pivot(index="Model", columns="Error Type", values="Proportion").fillna(0)
pivot = pivot[[e for e in error_order if e in pivot.columns]]  # 保证顺序

# 替换列名为可视化显示标签
pivot_display = pivot.rename(columns=label_mapping)

# 绘图
plt.figure(figsize=(14, 8))
pivot_display.plot(kind="barh", stacked=True, color=[color_map[k] for k in pivot.columns])

plt.xlabel("Proportion of Errors")
plt.title("Sentence-level Error Distribution by Model")
plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.15), ncol=3)
plt.tight_layout()
plt.savefig("/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/result/error_distribution_gapped.png", bbox_inches="tight")
plt.show()

