import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === 加载数据 ===
csv_path = "/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/all_type/caption/BLEU/bleu_by_model_and_charttype.csv"
df = pd.read_csv(csv_path)

# === 模型列表（顺序固定） ===
selected_models = [
    "InternLM-XC-v2-7B", "Qwen-VL-9.6B", "LLaVA-v1.5-13B",       # General MLLMs
    #"UniChart-201M", "Matcha-282M", "Pix2Struct-282M",       # Chart MLLMs
    "TinyChart-3B", #"MMCA-7B"
    , "ChartVLM-13B"
]
df = df[df['model'].isin(selected_models)]

# === 配色表（与你上传的颜色卡一致） ===
colors = {
    "InternLM-XC-v2-7B": "#F58787",
    "Qwen-VL-9.6B": "#FFD869",
    "TinyChart-3B": "#7CAEF0",
    "LLaVA-v1.5-13B": "#BB9393",
    "UniChart-201M": "#CDF296",
    "MMCA-7B": "#D1AEEC",
    "Pix2Struct-282M": "#8FB5DA",
    "Matcha-282M": "#FEAD76"
}

# === chart_type 列表 ===
chart_types = sorted(df['chart_type'].unique())
num_vars = len(chart_types)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # 闭合图形

# === 开始绘图 ===
plt.figure(figsize=(10, 8))
ax = plt.subplot(111, polar=True)

for model in selected_models:
    values = []
    for ct in chart_types:
        match = df[(df['model'] == model) & (df['chart_type'] == ct)]
        bleu = match['BLEU'].values[0] if not match.empty else 0
        values.append(bleu)
    values += values[:1]

    ax.plot(angles, values, label=model, linewidth=2, color=colors[model])
    ax.set_rlabel_position(0) 
    #ax.fill(angles, values, alpha=0.2, color=colors[model])

# === 设置标签 ===
ax.set_xticks(angles[:-1])
ax.set_xticklabels(chart_types, fontsize=11)
ax.set_title("BLEU Score by Model and Chart Type", fontsize=14)
plt.legend(loc='upper right', bbox_to_anchor=(1.4, 1.2), fontsize=13)
plt.tight_layout()
plt.savefig("/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/all_type/caption/BLEU/bleu_radar_chart_colored.png", dpi=300)
plt.show()
