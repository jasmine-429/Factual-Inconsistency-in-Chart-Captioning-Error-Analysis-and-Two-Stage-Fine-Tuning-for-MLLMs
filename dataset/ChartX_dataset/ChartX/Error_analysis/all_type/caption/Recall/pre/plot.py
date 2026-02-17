import pandas as pd
import matplotlib.pyplot as plt

# === 读取你的CSV ===
df = pd.read_csv("/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/all_type/caption/Recall/pre/value_precision_recall_by_model.csv")

# === 模型类别映射 ===
categories = {
    "General MLLM": ["InternLM-XC-v2-7B", "LLaVA-v1.5-13B", "Qwen-VL-9.6B"],
    "Chart MLLM": ["ChartInstruct-T5-3B", "ChartVLM", "MMCA-7B"],
    "Chart-specific": ["UniChart-201M", "Matcha-282M", "Pix2Struct-282M"]
}

# === 计算每类平均 Recall 和 Precision ===
results = []
for cat, models in categories.items():
    subset = df[df["model"].isin(models)]
    recall_mean = subset["number_recall"].mean()
    precision_mean = subset["number_precision"].mean()
    results.append([cat, recall_mean, precision_mean])

df_grouped = pd.DataFrame(results, columns=["Category", "Recall", "Precision"])

# === 绘制分组柱状图 ===
x = range(len(df_grouped))
bar_width = 0.35

plt.figure(figsize=(8,6))
plt.bar([i - bar_width/2 for i in x], df_grouped["Recall"], 
        width=bar_width, label="Recall", color="skyblue")
plt.bar([i + bar_width/2 for i in x], df_grouped["Precision"], 
        width=bar_width, label="Precision", color="orange")

# 数值标签
for i, val in enumerate(df_grouped["Recall"]):
    plt.text(i - bar_width/2, val + 0.5, f"{val:.1f}%", ha="center", fontsize=9)
for i, val in enumerate(df_grouped["Precision"]):
    plt.text(i + bar_width/2, val + 0.5, f"{val:.1f}%", ha="center", fontsize=9)

plt.xticks(x, df_grouped["Category"])
plt.ylabel("Percentage (%)")
plt.legend()
plt.tight_layout()
# 保存图片到当前工作目录
plt.savefig("/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/all_type/caption/Recall/pre/recall_precision_by_category.png", dpi=300, bbox_inches="tight")
plt.show()
