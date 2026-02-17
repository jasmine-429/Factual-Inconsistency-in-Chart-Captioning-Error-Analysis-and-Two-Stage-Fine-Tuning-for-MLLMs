import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# ========== 路径配置 ==========
input_path = "/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/merged_output.jsonl"
output_csv = "/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/result_eachsentence/model_error_distribution_real_proportion.csv"
output_fig = "/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/result_eachsentence/error_distribution_real.png"

# ========== 加载数据 ==========
records = []
with open(input_path, 'r') as f:
    for line in tqdm(f, desc="Parsing"):
        item = json.loads(line)
        model = item.get("model_name", "unknown")
        labels = item.get("labels", [])
        for sent_idx, label_list in enumerate(labels):
            for err in label_list:
                records.append({
                    "Model": model,
                    "Sentence": sent_idx,
                    "Error Type": err
                })

df = pd.DataFrame(records)

# ========== 统计每类错误在每模型下占生成句子的比例 ==========
sent_counts = []
with open(input_path, 'r') as f:
    for line in f:
        d = json.loads(line)
        model = d.get("model_name", "unknown")
        n_sent = len(d.get("labels", []))
        sent_counts.append((model, n_sent))

df_total = pd.DataFrame(sent_counts, columns=["Model", "NumSentences"]).groupby("Model").sum().reset_index()
df_error = df.groupby(["Model", "Error Type"]).size().reset_index(name="Count")
df_merged = pd.merge(df_error, df_total, on="Model")
df_merged["Proportion"] = df_merged["Count"] / df_merged["NumSentences"]
df_merged.to_csv(output_csv, index=False)

# ========== 可视化配置 ==========
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

# ========== 创建透视表 ==========
pivot = df_merged.pivot(index="Model", columns="Error Type", values="Proportion").fillna(0)
#pivot = pivot[[e for e in error_order if e in pivot.columns]]  # 保证顺序
#models = pivot.index.tolist()
#y_pos = np.arange(len(models))


# 使用预定义的模型顺序重新排序
model_order = [
    "InternLM-XC-v2-7B", "Qwen-VL-9.6B", "LLaVA-v1.5",
    "UniChart-201M", "Matcha-282M", "Pix2Struct-282M", 
    "TinyChart-3B", "MMCA-7B", "ChartVLM-13B"
]

# 重新排序 pivot DataFrame 的索引
pivot = df_merged.pivot(index="Model", columns="Error Type", values="Proportion").fillna(0)
pivot = pivot.reindex(model_order).fillna(0)

models = model_order
y_pos = np.arange(len(models))

print("模型名在数据中实际为：", df_merged["Model"].unique())
# ========== 绘图 ==========
fig, ax = plt.subplots(figsize=(14, 8))
left = np.zeros(len(models))
bar_height = 0.8

for error in error_order:
    if error in pivot.columns:
        values = pivot[error].values
        bars = ax.barh(
            y_pos,
            values,
            height=bar_height,
            left=left,
            color=color_map[error],
            label=label_mapping[error]
        )

        # 显式绘制白色竖线作为间隔（只画非零长度间隔）
        for i in range(len(y_pos)):
            if values[i] > 0:
                ax.plot(
                    [left[i], left[i]],
                    [y_pos[i] - bar_height / 2, y_pos[i] + bar_height / 2],
                    color="white",
                    linewidth=1
                )

        left += values

# ========== 轴和图例 ==========
ax.xaxis.grid(True, linestyle='--', alpha=0.6)
ax.set_yticks(y_pos)
ax.set_yticklabels(models)
ax.set_xlabel("Error Proportion (per sentence)")
ax.set_title("Real Sentence-Level Error Distribution by Model")

ax.invert_yaxis()

handles, labels = ax.get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="lower center", bbox_to_anchor=(0.5, -0.05),
    ncol=3, frameon=False
)

plt.tight_layout()
plt.subplots_adjust(bottom=0.2)
plt.savefig(output_fig, bbox_inches="tight")
plt.show()
