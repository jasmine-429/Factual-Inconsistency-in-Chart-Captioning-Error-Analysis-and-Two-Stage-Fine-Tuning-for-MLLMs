import json
import os
from bert_score import score
import pandas as pd
from collections import defaultdict

# ===== 配置路径 =====
input_json_list = [
    "/data/jguo376/project/model/TinyChart/chartx_caption/two_ft/two_ft.json"
]

annotation_path = "/data/jguo376/project/dataset/ChartX_dataset/ChartX/ChartX_annotation.json"

# ===== 读取人工标注，构建以 "img" 为 key 的索引 =====
with open(annotation_path, "r", encoding="utf-8") as f:
    annotation_list = json.load(f)
    annotation_dict = {item["img"]: item for item in annotation_list}

# ===== 读取预测数据并计算 BERTScore =====
overall_results = []
charttype_results = []

for path in input_json_list:
    with open(path, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    if not predictions or "model_name" not in predictions[0]:
        model_name = os.path.basename(path).replace("_caption_output.json", "")
    else:
        model_name = predictions[0]["model_name"]

    pred_texts_all, ref_texts_all = [], []
    charttype_to_preds = defaultdict(list)
    charttype_to_refs = defaultdict(list)

    for item in predictions:
        img_id = item.get("img")
        pred_caption = item.get("generated_caption", "").strip()
        gt_entry = annotation_dict.get(img_id, {})
        gt_caption = gt_entry.get("description", {}).get("output", "").strip()
        chart_type = gt_entry.get("chart_type", "unknown")

        if pred_caption and gt_caption:
            pred_texts_all.append(pred_caption)
            ref_texts_all.append(gt_caption)
            charttype_to_preds[chart_type].append(pred_caption)
            charttype_to_refs[chart_type].append(gt_caption)

    print(f"Calculating BERTScore for {model_name} with {len(pred_texts_all)} samples...")
    P, R, F1 = score(pred_texts_all, ref_texts_all, lang="en", verbose=True)
    overall_results.append({
        "model": model_name,
        "BERTScore_P": P.mean().item(),
        "BERTScore_R": R.mean().item(),
        "BERTScore_F1": F1.mean().item()
    })

    for chart_type in charttype_to_preds:
        preds = charttype_to_preds[chart_type]
        refs = charttype_to_refs[chart_type]
        if preds and refs:
            P_ct, R_ct, F1_ct = score(preds, refs, lang="en", verbose=False)
            charttype_results.append({
                "model": model_name,
                "chart_type": chart_type,
                "BERTScore_P": P_ct.mean().item(),
                "BERTScore_R": R_ct.mean().item(),
                "BERTScore_F1": F1_ct.mean().item()
            })

# ===== 保存结果 =====
df_overall = pd.DataFrame(overall_results)
df_charttype = pd.DataFrame(charttype_results)

df_overall.to_csv("/data/jguo376/project/model/TinyChart/chartx_caption/two_ft/bert/tiny_score_overall.csv", index=False)
df_charttype.to_csv("/data/jguo376/project/model/TinyChart/chartx_caption/two_ft/bert/tiny_score_by_charttype.csv", index=False)

print("✅ Done. Saved BERTScore results to CSV.")
