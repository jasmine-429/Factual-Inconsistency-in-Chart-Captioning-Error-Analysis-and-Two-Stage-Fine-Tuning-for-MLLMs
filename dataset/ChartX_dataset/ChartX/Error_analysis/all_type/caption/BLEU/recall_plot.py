import pandas as pd
import matplotlib.pyplot as plt

# 读取 CSV 文件（替换为你的实际路径）
csv_path = "/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/all_type/caption/Recall/value_recall_by_model.csv"
df = pd.read_csv(csv_path)

# 模型排序与分组标签
model_order = [
    "InternLM-XC-v2-7B", "Qwen-VL-9.6B", "LLaVA-v1.5-13B",       # General MLLMs
    #"UniChart-201M", "Matcha-282M", "Pix2Struct-282M",           # Chart MLLMs
    "TinyChart-3B"#, "MMCA-7B"
    , "ChartVLM"                        # Specialist Chart Models
]
group_labels = ["General\nMLLMs", #"Specialist\nChart Models", 
"Chart\nMLLMs"]
group_pos = [1, 4, 7]

# 每个模型的颜色（自定义）
colors = [
    "#a2d2e7", "#67a8cd", "#3581b7",  # General
    #"#b3e19b", "#6fb3a8", "#50aa4b",  # Chart
    "#cdb6da", #"#9a7fbd", 
    "#704ba3"   # Specialist
]

# 排序并格式化
# === 只保留指定的模型行 ===
df = df[df['model'].isin(model_order)].copy()

# 排序并格式化
df['model'] = pd.Categorical(df['model'], categories=model_order, ordered=True)
df = df.sort_values("model").reset_index(drop=True)
df['model'] = df['model'].astype(str)

# 绘图
plt.figure(figsize=(8, 6))
bars = plt.barh(df['model'], df['avg_recall'], color=colors)
plt.axhline(y=2.5, color='gray', linestyle='--', linewidth=1)
# 添加数值标签
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.8, bar.get_y() + bar.get_height() / 2,
             f'{width:.2f}', va='center', fontsize=9)

# 添加分组文字
for i, label in enumerate(group_labels):
    y_pos = group_pos[i]
    plt.text(-10, y_pos, label, va='center', ha='right',
             fontsize=10, fontweight='bold')

#plt.legend(handles=legend_elements, loc="upper right")

# 样式配置
plt.xlabel("Average Recall (%)")
plt.xlim(0, 38)
plt.gca().invert_yaxis()
plt.tight_layout()

# 保存图像
plt.savefig("/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/all_type/caption/BLEU/output/avg_recall_by_model.png", dpi=300, bbox_inches='tight')  # 可改为 pdf/svg 等格式
plt.show()
