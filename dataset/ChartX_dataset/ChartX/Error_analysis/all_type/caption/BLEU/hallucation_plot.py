import pandas as pd
import matplotlib.pyplot as plt

# ===== 配置 =====
csv_path = "/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/all_type/caption/BLEU/hallucination_stats_by_model.csv"
model_order = [
    "InternLM-XC-v2-7B", "Qwen-VL-9.6B", "LLaVA-v1.5-13B",       # General MLLMs
    "UniChart-201M", "Matcha-282M", "Pix2Struct-282M",
    "ChartInstruct-T5-3B", "MMCA-7B", "ChartVLM-13B"
]

# ===== 加载并预处理数据 =====
df = pd.read_csv(csv_path)
df["model_name"] = pd.Categorical(df["model_name"], categories=model_order, ordered=True)
df = df.sort_values("model_name")

# ===== 计算幻觉百分比 =====
df["year_percent"] = df["year_hallucination"] / df["total"] * 100
df["country_percent"] = df["country_hallucination"] / df["total"] * 100
df["overall_percent"] = df["overall_hallucination"] / df["total"] * 100

# ===== 绘图（用整数索引避免 float 问题） =====
x = range(len(df))
labels = df["model_name"].astype(str).tolist()

plt.figure(figsize=(10, 6))

plt.plot(x, df["year_percent"],
         marker='o', markersize=8, color="#1f77b4", linewidth=2, label="Year (OOC)")

plt.plot(x, df["country_percent"],
         marker='^', markersize=8, color="#ff7f0e", linewidth=2, label="Country (OOC)")

plt.plot(x, df["overall_percent"],
         marker='s', markersize=8, color="#2ca02c", linewidth=2, label="Overall OCC (Year or Country)")

plt.xticks(x, labels, rotation=45, ha="right")
plt.ylabel("Hallucination Rate (%)")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/all_type/caption/BLEU/hallucination_chart.png", dpi=300)
plt.show()
