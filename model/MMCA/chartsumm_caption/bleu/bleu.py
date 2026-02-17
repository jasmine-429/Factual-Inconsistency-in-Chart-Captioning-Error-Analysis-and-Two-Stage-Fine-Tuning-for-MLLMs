import os
import json
import pandas as pd
import sacrebleu

# ========= 路径配置（按需修改）=========
orig_files = {
    "valid_k": "/data/jguo376/project/dataset/chartsumm/test_k.json",
    "valid_s": "/data/jguo376/project/dataset/chartsumm/test_s.json",
}
pred_files = {
    "valid_k": "/data/jguo376/project/model/MMCA/chartsumm_caption/test_k_output.json",
    "valid_s": "/data/jguo376/project/model/MMCA/chartsumm_caption/test_s_output.json",
}
out_dir = "/data/jguo376/project/model/MMCA/chartsumm_caption/bleu"
os.makedirs(out_dir, exist_ok=True)

def load_refs(path):
    """读取原始验证集，返回 {image: summary}"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    image2ref = {}
    for item in data:
        img = str(item.get("image", "")).strip()
        ref = (item.get("summary") or "").strip()
        if img and ref:
            image2ref[img] = ref
    return image2ref

def load_preds(path):
    """读取模型输出（image + generated_caption）"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for it in data:
        img = str(it.get("image", "")).strip()
        hyp = (it.get("generated_caption") or "").strip()
        if img and hyp:
            out.append({"image": img, "generated_caption": hyp})
    return out

def compute_bleu_for_set(tag, refs_dict, preds_list):
    """
    计算单个集合的 BLEU：
      - corpus BLEU（整体）
      - sentence BLEU（逐样本，方便排查）
    保存 detail 与 summary 两个 CSV，返回 (preds_texts, refs_texts) 以便 overall 汇总
    """
    preds_texts, refs_texts, keep_images, sent_bleus = [], [], [], []

    for it in preds_list:
        img = it["image"]
        hyp = it["generated_caption"]
        ref = refs_dict.get(img, "")
        if not ref:
            continue
        preds_texts.append(hyp)
        refs_texts.append(ref)
        keep_images.append(img)

        # sentence BLEU（可选）
        sb = sacrebleu.sentence_bleu(hyp, [ref]).score
        sent_bleus.append(sb)

    if not preds_texts:
        print(f"[{tag}] ⚠️ 无可对齐样本，跳过。")
        return [], []

    # corpus BLEU
    c_bleu = sacrebleu.corpus_bleu(preds_texts, [refs_texts])

    # 明细
    df_detail = pd.DataFrame({
        "image": keep_images,
        "pred": preds_texts,
        "ref": refs_texts,
        "sentence_BLEU": sent_bleus
    })
    detail_csv = os.path.join(out_dir, f"{tag}_bleu_detail.csv")
    df_detail.to_csv(detail_csv, index=False)

    # 汇总
    df_sum = pd.DataFrame([{
        "set": tag,
        "num_samples": len(preds_texts),
        "corpus_BLEU": round(c_bleu.score, 2)
    }])
    summary_csv = os.path.join(out_dir, f"{tag}_bleu_summary.csv")
    df_sum.to_csv(summary_csv, index=False)

    print(f"[{tag}] ✅ corpus BLEU: {round(c_bleu.score, 2)} | 样本数: {len(preds_texts)}")
    print(f"[{tag}] 📄 明细:   {detail_csv}")
    print(f"[{tag}] 📄 汇总:   {summary_csv}")
    return preds_texts, refs_texts

# ========= 主流程 =========
all_preds, all_refs = [], []

for tag in ["valid_k", "valid_s"]:
    refs_dict = load_refs(orig_files[tag])
    preds_list = load_preds(pred_files[tag])
    p, r = compute_bleu_for_set(tag, refs_dict, preds_list)
    all_preds.extend(p)
    all_refs.extend(r)

# overall（两个集合合并）
if all_preds:
    c_bleu_all = sacrebleu.corpus_bleu(all_preds, [all_refs])
    df_overall = pd.DataFrame([{
        "set": "combined_valid_k_s",
        "num_samples": len(all_preds),
        "corpus_BLEU": round(c_bleu_all.score, 2)
    }])
    overall_csv = os.path.join(out_dir, "overall_bleu_summary.csv")
    df_overall.to_csv(overall_csv, index=False)
    print(f"[overall] ✅ corpus BLEU: {round(c_bleu_all.score, 2)} | 样本数: {len(all_preds)}")
    print(f"[overall] 📄 汇总:   {overall_csv}")
else:
    print("❗ overall：无可对齐样本。")


