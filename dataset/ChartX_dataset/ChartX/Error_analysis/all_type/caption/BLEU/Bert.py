import json
import os
from bert_score import score
import pandas as pd

# ===== 配置路径 =====
input_json_list = [
    "/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/all_type/caption/deep_analysis/caption_all/internlm_caption_output_org.json",
    "/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/all_type/caption/deep_analysis/caption_all/llava_caption_output.json",
    "/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/all_type/caption/deep_analysis/caption_all/matcha_caption_output.json",
    "/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/all_type/caption/deep_analysis/caption_all/mmca_caption_output.json",
    "/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/all_type/caption/deep_analysis/caption_all/pix2struct_caption_output.json",
    "/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/all_type/caption/deep_analysis/caption_all/qwen_caption_output.json",
    "/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/all_type/caption/deep_analysis/caption_all/tinychart_caption_output.json",
    "/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/all_type/caption/deep_analysis/caption_all/unichart_caption_output.json"
]

annotation_path = "/data/jguo376/project/dataset/ChartX_dataset/ChartX/ChartX_annotation.json"

# ===== 读取人工标注，构建以 "img" 为 key 的索引 =====
with open(annotation_path, "r", encoding="utf-8") as f:
    annotation_list = json.load(f)
    annotation_dict = {item["img"]: item for item in annotation_list}

# ===== 读取预测数据并计算 BERTScore =====
results = []

for path in input_json_list:
    model_name = os.path.basename(path).replace("_caption_output.json", "")
    with open(path, "r", encoding="utf-8") as f:
        predictions = json.load(f)
    
    pred_texts, ref_texts = [], []
    for item in predictions:
        img_id = item.get("img")
        pred_caption = item.get("generated_caption", "").strip()
        gt_caption = annotation_dict.get(img_id, {}).get("description", {}).get("output", "").strip()
        
        if pred_caption and gt_caption:
            pred_texts.append(pred_caption)
            ref_texts.append(gt_caption)

    print(f"Calculating BERTScore for {model_name} with {len(pred_texts)} samples...")
    P, R, F1 = score(pred_texts, ref_texts, lang="en", verbose=True)
    results.append({
        "model": model_name,
        "BERTScore_P": P.mean().item(),
        "BERTScore_R": R.mean().item(),
        "BERTScore_F1": F1.mean().item()
    })

# ===== 保存结果 =====
df = pd.DataFrame(results)
df.to_csv("/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/all_type/caption/BLEU/bert_score_results.csv", index=False)
print("✅ Saved BERTScore results to CSV.")
