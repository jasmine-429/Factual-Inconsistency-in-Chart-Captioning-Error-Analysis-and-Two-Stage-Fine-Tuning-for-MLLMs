import json
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# ======= 路径配置 =======
input_path = "/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/merged_output.jsonl"
output_csv = "/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/result/model_error_distribution.csv"
output_png = "/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/result/error_distribution_gapped.png"

# ======= 1. 加载数据并解析错误类型 =======
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

df = pd.DataFrame(records)

# ======= 2. 统计各类错误比例 =======
error_counts = df.groupby(["Model", "Error Type"]).size().reset_index(name="Count")
model_totals = df.groupby("Model").size().reset_index(name="Total")
result_df = pd.merge(error_counts, model_totals, on="Model")
result_df["Proportion"] = result_df["Count"] / result_df["Total"]

# 保存 CSV
result_df.to_csv(output_csv, index=False)
print("📊 Error distribution saved to:", output_csv)

# ======= 3. 标签映射与颜色 =======
label_mapping = {
    "value_error": "Value Error",
    "label_error": "Label Error",
    "trend_error": "Trend Error",
    "ooc_error": "Out Of Context Error",
    "magnitude_error": "Magnitude Error",
    "nonsense_error": "Nonsense Error"
}

error_order = list(label_mapping.keys())
color_map = {
    "value_error": "#1f77b4",
    "label_error": "#ff7f0e",
    "trend_error": "#d62728",
    "ooc_error": "#2ca02c",
    "magnitude_error": "#9467bd",
    "nonsense_error": "#c49c94"
}

# ======= 4. 创建透视表 =======
#pivot = result_df.pivot(index="Model", columns="Error Type", values="Proportion").fillna(0)
#pivot = pivot[[e for e in error_order if e in pivot.columns]]

#models = pivot.index.tolist()
#y_pos = np.arange(len(models))
model_order = [
    "InternLM-XC-v2-7B", "Qwen-VL-9.6B" "LLaVA-v1.5"
    "UniChart-201M", "Matcha-282M", "Pix2Struct-282M", 
    "TinyChart-3B", "MMCA-7B", "ChartVLM-13B"
]

# 重新排序 pivot DataFrame 的索引
pivot = result_df.pivot(index="Model", columns="Error Type", values="Proportion").fillna(0)
pivot = pivot.reindex(model_order).fillna(0)

models = model_order
y_pos = np.arange(len(models))

# ======= 5. 绘图（添加白线分隔）=======
fig, ax = plt.subplots(figsize=(14, 8))
bar_height = 0.7
left = np.zeros(len(models))

for error in error_order:
    if error in pivot.columns:
        values = pivot[error].values

        # 先画主 bar
        ax.barh(
            y_pos,
            values,
            height=bar_height,
            left=left,
            color=color_map[error],
            label=label_mapping[error]
        )

        # 添加白色竖线分隔（仅在长度非零时绘制）
        for i, v in enumerate(values):
            if v > 0:
                ax.plot(
                    [left[i], left[i]],
                    [y_pos[i] - bar_height / 2, y_pos[i] + bar_height / 2],
                    color='white',
                    linewidth=1
                )

        left += values  # 更新下一段起点

# ======= 6. 图标设置 =======
ax.xaxis.grid(True, linestyle='--', alpha=0.6)
ax.set_yticks(y_pos)
ax.set_yticklabels(models, fontsize=12)
ax.set_xlabel("Proportion of Errors", fontsize=12)
ax.set_title("Error Distribution by Model", fontsize=14)
ax.invert_yaxis()

handles, labels = ax.get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.1),
    ncol=3,
    frameon=False,
    fontsize=11
)

plt.tight_layout()
plt.subplots_adjust(bottom=0.25)
plt.savefig(output_png, bbox_inches="tight", dpi=300)
print("📈 Plot saved to:", output_png)
plt.show()
