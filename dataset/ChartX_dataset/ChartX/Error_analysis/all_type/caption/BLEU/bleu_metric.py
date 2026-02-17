import os
import json
import pandas as pd
import sacrebleu
from collections import defaultdict

# === 路径配置 ===
annotation_path = "/data/jguo376/project/dataset/ChartX_dataset/ChartX/ChartX_annotation.json"
model_output_dir = "/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/all_type/caption/deep_analysis/caption_all"
output_dir = "/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/all_type/caption/BLEU"
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

# === 分别处理每个模型输出 ===
by_model = defaultdict(lambda: {"refs": [], "preds": []})
by_model_chart = defaultdict(lambda: {"refs": [], "preds": []})

for filename in os.listdir(model_output_dir):
    if not filename.endswith(".json"):
        continue

    model_path = os.path.join(model_output_dir, filename)
    with open(model_path, "r", encoding="utf-8") as f:
        model_data = json.load(f)

    for item in model_data:
        relative_img = item.get("img", "").strip().lstrip("./")
        ref = img2ref.get(relative_img, "").strip()
        pred = item.get("generated_caption", "").strip()
        model = item.get("model_name", filename.replace(".json", ""))
        chart_type = item.get("chart_type", "unknown")

        if ref and pred:
            by_model[model]["refs"].append(ref)
            by_model[model]["preds"].append(pred)
            by_model_chart[(model, chart_type)]["refs"].append(ref)
            by_model_chart[(model, chart_type)]["preds"].append(pred)

# === 计算 BLEU：模型整体 ===
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
if not df_model.empty:
    df_model = df_model.sort_values(by="BLEU", ascending=False)
    df_model.to_csv(os.path.join(output_dir, "bleu_by_model.csv"), index=False)
    print("\n📊 BLEU by model:")
    print(df_model.to_string(index=False))
else:
    print("❌ 没有任何模型生成的 caption 匹配参考答案，无法计算 BLEU。")

# === 计算 BLEU：模型 × chart_type ===
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
if not df_chart.empty:
    df_chart = df_chart.sort_values(by=["model", "chart_type"])
    df_chart.to_csv(os.path.join(output_dir, "bleu_by_model_and_charttype.csv"), index=False)
    print("\n📊 BLEU by model + chart_type:")
    print(df_chart.to_string(index=False))
else:
    print("❌ 没有可用的模型 + chart_type 样本用于 BLEU 计算。")
