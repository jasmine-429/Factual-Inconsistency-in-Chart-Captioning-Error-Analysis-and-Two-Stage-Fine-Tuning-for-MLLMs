import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({
    "font.size": 15,
    "legend.fontsize": 15,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})

# === 加载数据 ===
csv_path = "/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/all_type/caption/BLEU/bert_score_results_by_charttype.csv"
df = pd.read_csv(csv_path)

# === 配色表（保持一致）===
colors = {
    "InternLM-XC-v2-7B": "#F58787",
    "Qwen-VL-9.6B": "#FFD869",
    "TinyChart-3B": "#7CAEF0",
    "LLaVA-v1.5-13B": "#BB9393",
    "UniChart-201M": "#CDF296",
    "MMCA-7B": "#D1AEEC",
    "Pix2Struct-282M": "#8FB5DA",
    "Matcha-282M": "#FEAD76",
    "ChartVLM-13B": "#74C476"
}

chart_types = sorted(df['chart_type'].unique())
angles = np.linspace(0, 2 * np.pi, len(chart_types), endpoint=False).tolist()
angles += angles[:1]

# === 分组模型 ===
main_models = [m for m in colors if m != "Matcha-282M"]
matcha_model = ["Matcha-282M"]

def plot_models(ax, model_list, ylim):
    for model in model_list:
        values = []
        for ct in chart_types:
            match = df[(df['model'] == model) & (df['chart_type'] == ct)]
            score = match['BERTScore_F1'].values[0] if not match.empty else np.nan
            values.append(score)
        values += values[:1]
        ax.plot(angles, values, label=model, linewidth=2, color=colors[model])
        ax.fill(angles, values, alpha=0.05, color=colors[model])
    
    ax.set_ylim(*ylim)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(chart_types, fontsize=16)
    
    yticks = np.linspace(ylim[0], ylim[1], 3)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{v:.2f}" for v in yticks], fontsize=14)
    ax.tick_params(axis='y', pad=-10)
    ax.set_rlabel_position(135)
    #ax.set_title(title, fontsize=14)

# === 开始绘图 ===
fig, axs = plt.subplots(1, 2, subplot_kw=dict(polar=True), figsize=(18, 8))

# 主图（不含 Matcha）
plot_models(axs[0], main_models, (0.83, 0.93))

# Matcha 单独图
plot_models(axs[1], matcha_model, (0.70, 0.86))

# 图例
# === 收集所有图例元素并合并 ===
handles_labels = []
for ax in axs:
    handles, labels = ax.get_legend_handles_labels()
    handles_labels.append((handles, labels))

all_handles = sum([h for h, _ in handles_labels], [])
all_labels = sum([l for _, l in handles_labels], [])

# === 设置图例在两个图之间 ===
fig.legend(
    all_handles, all_labels,
    loc='center',              # 图例居中显示
    bbox_to_anchor=(0.52, 0.82), # 坐标相对于整个 figure 空间
    ncol=1,                    # 竖直排列
    fontsize=15,
    frameon=False
)


# 替代 plt.tight_layout()
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/all_type/caption/BLEU/bert_radar_matcha_separated.png", dpi=300)
plt.show()
