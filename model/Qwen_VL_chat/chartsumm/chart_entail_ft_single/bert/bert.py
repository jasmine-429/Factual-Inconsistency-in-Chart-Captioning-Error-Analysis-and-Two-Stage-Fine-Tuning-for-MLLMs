import os
import json
import pandas as pd
from bert_score import score

# ========= 路径配置（按需修改）=========
# 原始（带 reference 的）验证集
orig_files = {
    "test_k": "/data/jguo376/project/dataset/chartsumm/test_k.json",
    "test_s": "/data/jguo376/project/dataset/chartsumm/test_s.json",
}
# 模型生成结果（你的推理输出，只有 image + generated_caption）
pred_files = {
    "test_k": "/data/jguo376/project/model/Qwen_VL_chat/chartsumm/chart_entail_ft_single/test_k_output.json",
    "test_s": "/data/jguo376/project/model/Qwen_VL_chat/chartsumm/chart_entail_ft_single/test_s_output.json",
}

out_dir = "/data/jguo376/project/model/Qwen_VL_chat/chartsumm/chart_entail_ft_single/bert"
os.makedirs(out_dir, exist_ok=True)

def load_refs(path):
    """读取原始验证集，返回 {image: summary}"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    image2ref = {}
    for item in data:
        img = item.get("image", "").strip()
        ref = (item.get("summary") or "").strip()
        if img and ref:
            image2ref[img] = ref
    return image2ref

def load_preds(path):
    """读取模型输出，返回列表 [{'image':..., 'generated_caption':...}, ...]"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 兼容可能的 dict/tuple 混入
    out = []
    for it in data:
        img = str(it.get("image", "")).strip()
        cap = (it.get("generated_caption") or "").strip()
        if img and cap:
            out.append({"image": img, "generated_caption": cap})
    return out

def compute_bertscore_single_set(tag, preds, image2ref, out_dir):
    """
    对单个文件（一个验证集）计算 BERTScore，并存两份CSV：
    1) 明细（每条样本的 F1）  2) 汇总（P/R/F1 均值）
    返回用于 overall 汇总的 (pred_texts, ref_texts)
    """
    pair_list = []
    pred_texts, ref_texts, keep_images = [], [], []
    for it in preds:
        img = it["image"]
        pred = it["generated_caption"]
        ref = image2ref.get(img, "")
        if not ref:
            continue
        pred_texts.append(pred)
        ref_texts.append(ref)
        keep_images.append(img)

    if len(pred_texts) == 0:
        print(f"[{tag}] 没有可对齐的样本，跳过。")
        return [], []

    # 计算 BERTScore
    P, R, F1 = score(pred_texts, ref_texts, lang="en", verbose=True)

    # 保存逐样本明细
    df_detail = pd.DataFrame({
        "image": keep_images,
        "pred": pred_texts,
        "ref": ref_texts,
        "BERTScore_P": P.tolist(),
        "BERTScore_R": R.tolist(),
        "BERTScore_F1": F1.tolist(),
    })
    detail_csv = os.path.join(out_dir, f"{tag}_bertscore_detail.csv")
    df_detail.to_csv(detail_csv, index=False)

    # 保存本集合汇总
    df_sum = pd.DataFrame([{
        "set": tag,
        "num_samples": len(pred_texts),
        "BERTScore_P": float(P.mean()),
        "BERTScore_R": float(R.mean()),
        "BERTScore_F1": float(F1.mean()),
    }])
    summary_csv = os.path.join(out_dir, f"{tag}_bertscore_summary.csv")
    df_sum.to_csv(summary_csv, index=False)

    print(f"[{tag}] ✅ 明细: {detail_csv}")
    print(f"[{tag}] ✅ 汇总: {summary_csv}")

    return pred_texts, ref_texts

# ========= 主流程 =========
all_preds_texts, all_refs_texts = [], []
all_sets_summary = []

for tag in ["test_k", "test_s"]:
    # 加载 refs & preds
    image2ref = load_refs(orig_files[tag])
    preds = load_preds(pred_files[tag])

    # 计算单集合分数并保存
    preds_texts, refs_texts = compute_bertscore_single_set(tag, preds, image2ref, out_dir)
    if preds_texts:
        all_preds_texts.extend(preds_texts)
        all_refs_texts.extend(refs_texts)

# ========= 合并两个集合后的 overall =========
if all_preds_texts:
    P_all, R_all, F1_all = score(all_preds_texts, all_refs_texts, lang="en", verbose=True)
    df_overall = pd.DataFrame([{
        "set": "combined_valid_k_s",
        "num_samples": len(all_preds_texts),
        "BERTScore_P": float(P_all.mean()),
        "BERTScore_R": float(R_all.mean()),
        "BERTScore_F1": float(F1_all.mean()),
    }])
    overall_csv = os.path.join(out_dir, "overall_bertscore_summary.csv")
    df_overall.to_csv(overall_csv, index=False)
    print(f"[overall] ✅ 汇总: {overall_csv}")
else:
    print("❗ 没有任何可对齐样本用于 overall 统计。")

