import os
import json
import pandas as pd
import sacrebleu
from collections import defaultdict

# === 路径配置 ===
annotation_path = "/data/jguo376/project/dataset/ChartX_dataset/ChartX/ChartX_annotation.json"
model_output_path = "/data/jguo376/project/model/chartInstruction/chartx_caption.json"
output_dir = "/data/jguo376/project/model/chartInstruction/bleu/chartx"
os.makedirs(output_dir, exist_ok=True)

# === 加载参考 captions ===
with open(annotation_path, "r", encoding="utf-8") as f:
    annotation_data = json.load(f)

img2ref = {}
for item in annotation_data:
    image_path = item.get("img", "").strip().lstrip("./")
    ref_caption = item.get("description", {}).get("output", "").strip()
    if image_path and ref_caption:
        img2ref[image_path] = ref_caption

print(f"✅ 加载参考图像数: {len(img2ref)}")

# === 加载模型预测 ===
with open(model_output_path, "r", encoding="utf-8") as f:
    model_data = json.load(f)

by_model = defaultdict(lambda: {"refs": [], "preds": []})
by_model_chart = defaultdict(lambda: {"refs": [], "preds": []})

for item in model_data:
    relative_img = item.get("img", "").strip().lstrip("./")
    ref = img2ref.get(relative_img, "").strip()
    pred = item.get("generated_caption", "").strip()
    model = item.get("model_name", "qwen")  # 或者写死为 "Qwen"
    chart_type = item.get("chart_type", "unknown")

    if ref and pred:
        by_model[model]["refs"].append(ref)
        by_model[model]["preds"].append(pred)
        by_model_chart[(model, chart_type)]["refs"].append(ref)
        by_model_chart[(model, chart_type)]["preds"].append(pred)

# === 计算 BLEU：整体 ===
model_rows = []
for model, g in by_model.items():
    if g["refs"]:
        bleu = sacrebleu.corpus_bleu(g["preds"], [g["refs"]])
        model_rows.append({
            "model": model,
            "BLEU": round(bleu.score, 2),
            "num_samples": len(g["preds"])
        })

df_model = pd.DataFrame(model_rows)
df_model.to_csv(os.path.join(output_dir, "bleu_overall.csv"), index=False)
print("\n📊 BLEU overall:")
print(df_model.to_string(index=False))

# === 计算 BLEU：按 chart_type ===
chart_rows = []
for (model, chart_type), g in by_model_chart.items():
    if g["refs"]:
        bleu = sacrebleu.corpus_bleu(g["preds"], [g["refs"]])
        chart_rows.append({
            "model": model,
            "chart_type": chart_type,
            "BLEU": round(bleu.score, 2),
            "num_samples": len(g["preds"])
        })

df_chart = pd.DataFrame(chart_rows)
df_chart.to_csv(os.path.join(output_dir, "bleu_by_charttype.csv"), index=False)
print("\n📊 BLEU by chart_type:")
print(df_chart.to_string(index=False))

