import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# ===== 配置 =====
model_order = [
    "InternLM-XC-v2-7B", "Qwen-VL-9.6B", "LLaVA-v1.5-13B",       # General MLLMs
    "UniChart-201M", "Matcha-282M", "Pix2Struct-282M",          # Specialist Chart Models 
    "ChartInstruct-T5-3B", "MMCA-7B", "ChartVLM-13B"                   # Chart MLLMs
]
group_map = {
    "InternLM-XC-v2-7B": "General MLLMs",
    "Qwen-VL-9.6B": "General MLLMs",
    "LLaVA-v1.5-13B": "General MLLMs",
    "UniChart-201M": "Specialist Chart Models",
    "Matcha-282M": "Specialist Chart Models",
    "Pix2Struct-282M": "Specialist Chart Models",
    "ChartInstruct-T5-3B": "Chart MLLMs",
    "MMCA-7B": "Chart MLLMs",
    "ChartVLM-13B": "Chart MLLMs"
}
color_map = {
    "value_error": "#F58787",
    "label_error": "#FEAD76",
    "trend_error": "#FFD869",
    "ooc_error": "#7CAEF0",
    "magnitude_error": "#D1AEEC",
    "nonsense_error": "#b3e19b"
}

# ===== Step 1: 读取文件 =====
with open("/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/all_type/all_typr.jsonl", "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

# ===== Step 2: 修复错误标签拼写 =====
for item in data:
    cleaned_labels = []
    for label_list in item.get("labels", []):
        corrected = ["ooc_error" if l == "ooc_label_errorerror" else l for l in label_list]
        cleaned_labels.append(corrected)
    item["labels"] = cleaned_labels

# ===== Step 3: 聚合错误数据 =====
group_chart_error = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
for item in data:
    model = item.get("model_name")
    chart_type = item.get("chart_type")
    label_lists = item.get("labels", [])
    
    if model in group_map and chart_type:
        group = group_map[model]
        for label_list in label_lists:
            for error_type in label_list:
                group_chart_error[group][chart_type][error_type] += 1

# ===== Step 4: 转换为 DataFrame =====
records = []
for group in group_chart_error:
    for chart_type in group_chart_error[group]:
        for error_type in group_chart_error[group][chart_type]:
            count = group_chart_error[group][chart_type][error_type]
            records.append({
                "Group": group,
                "Chart Type": chart_type,
                "Error Type": error_type,
                "Count": count
            })

df = pd.DataFrame(records)

# ===== Step 5: 画堆叠柱状图（带顶部白线分隔） =====
pivot_df = df.pivot_table(
    index=["Group", "Chart Type"],
    columns="Error Type",
    values="Count",
    fill_value=0
).reset_index()

error_types = list(color_map.keys())
groups = pivot_df["Group"].unique()


fig, axes = plt.subplots(nrows=len(groups), figsize=(16, 5 * len(groups)), sharex=True)
if len(groups) == 1:
    axes = [axes]

for ax, group in zip(axes, groups):
    sub_df = pivot_df[pivot_df["Group"] == group]
    chart_types = sub_df["Chart Type"]
    bottom = np.zeros(len(sub_df))
    
    for err in error_types:
        values = sub_df[err].values if err in sub_df.columns else np.zeros(len(sub_df))
        bars = ax.bar(chart_types, values, bottom=bottom, label=err, color=color_map.get(err, "#cccccc"))

        # 添加顶部横向白线
        for rect, h in zip(bars, values):
            if h > 0:
                left = rect.get_x()
                width = rect.get_width()
                top = rect.get_y() + rect.get_height()
                ax.plot(
                    [left, left + width],
                    [top, top],
                    color="white",
                    linewidth=1
                )

        bottom += values

    ax.set_title(group, fontsize=16)
    ax.set_ylabel("Error Count", fontsize=16)
    ax.set_xticks(np.arange(len(chart_types)))
    ax.set_xticklabels(chart_types, rotation=45, ha="right", fontsize=16)

axes[1].legend(
    title="Error Type",
    fontsize=14,
    title_fontsize=15,
    loc='lower left',
    bbox_to_anchor=(0.0, 0.6),  # ⬅️ 位置在图内左下方，稍微偏出图形范围
    ncol=2,
    frameon=True
)

plt.tight_layout()

# ===== Step 6: 保存文件 =====
df.to_csv("/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/all_type/chart_type_error_counts.csv", index=False)
fig.savefig("/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/all_type/chart_type_error_stacked_bar.png", dpi=300)

print("✅ 保存完成：chart_type_error_counts.csv + chart_type_error_stacked_bar.png")
