import sys
sys.path.append("/data/jguo376/project/model/TinyChart")

import torch
import json
import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.stats import kendalltau

from tinychart.model.builder import load_pretrained_model
from tinychart.mm_utils import get_model_name_from_path
from peft import PeftModel

# ========= 设置路径 =========
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

model_path = "/data/jguo376/pretrained_models/TinyChart-3B-768"
lora_path = "/data/jguo376/project/model/TinyChart/checkpoints/chart_entail/checkpoint-700"
use_lora = True

data_path = "/data/jguo376/project/dataset/test_dataset/ChartX/test_data/dataset/combined_dataset.jsonl"
save_dir = "/data/jguo376/project/model/TinyChart/chart_entail/myself/prompt_700"
os.makedirs(save_dir, exist_ok=True)

FACTUAL_ERROR_TYPES = ['label_error', 'magnitude_error', 'ooc_error', 'trend_error', 'value_error', 'nonsense_error']

# ========= 加载模型 =========
print("Loading model...")
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path),
    device="cuda:0"
)

if use_lora:
    print("Loading LoRA...")
    model = PeftModel.from_pretrained(model, lora_path)
    print("Merging LoRA weights...")
    model = model.merge_and_unload()
    print("LoRA weights loaded and merged.")

model = model.half()
print(f"Model loaded on device: {next(model.parameters()).device}")
print(f"Model dtype: {next(model.parameters()).dtype}")

# ========= 数据预处理 =========
def format_query(sentence):
    return f"Does the image entail this statement: \"{sentence}\"? Answer with 'Yes' or 'No'."

def proccess_samples(samples):
    processed = []
    for sample in samples:
        caption_label = int(all([label not in FACTUAL_ERROR_TYPES for sent_labels in sample["labels"] for label in sent_labels]))
        for sentence, sent_labels in zip(sample["generated_caption"], sample["labels"]):
            image_path = os.path.join("/data/jguo376/project/dataset/ChartX_dataset/ChartX", sample["img"].lstrip("./"))
            query = format_query(sentence)
            sent_label = 0 if any([l in FACTUAL_ERROR_TYPES for l in sent_labels]) else 1
            row = [sample['_id'], image_path, query, sent_label, caption_label]
            processed.append(row)
    return pd.DataFrame(processed, columns=['_id','image_path','prompt','sent_label','caption_label'])

# ========= 推理函数 =========
def get_prediction(processed_df, tokenizer, model, image_processor):
    from tinychart.conversation import conv_templates

    binary_positive_probs = []
    yes_id = tokenizer.convert_tokens_to_ids("yes")
    no_id = tokenizer.convert_tokens_to_ids("no")

    for row in tqdm(processed_df.itertuples(), total=len(processed_df)):
        try:
            image = Image.open(row.image_path).convert("RGB")
            prompt = row.prompt
            conv = conv_templates["phi"].copy()
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            text = conv.get_prompt()

            inputs = tokenizer([text], return_tensors="pt").to("cuda")
            image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0].unsqueeze(0)
            image_tensor = image_tensor.to("cuda").to(model.dtype)

            with torch.no_grad():
                outputs = model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    images=image_tensor,
                    return_dict=True,
                )
                logits = outputs.logits
                next_token_logits = logits[0, -1]

            probs = torch.softmax(next_token_logits[[yes_id, no_id]], dim=0)
            entail_prob = probs[0].item()
            binary_positive_probs.append(entail_prob)

        except Exception as e:
            print(f"Error processing row {row.Index}: {e}")
            binary_positive_probs.append(0.5)

    processed_df["binary_entailment_prob"] = binary_positive_probs
    return processed_df

# ========= 开始执行 =========
print("Loading data...")
with open(data_path, "r") as f:
    chocolate = [json.loads(line) for line in f]

processed = proccess_samples(chocolate)
processed = get_prediction(processed, tokenizer, model, image_processor)

# ========= 使用样本 split 字段分类（summarization/description） =========
_id_to_split = {sample["_id"]: sample["split"] for sample in chocolate}
processed["split"] = processed["_id"].map(_id_to_split)

id2score = processed.groupby('_id')['binary_entailment_prob'].min().to_dict()
processed["chartve_score"] = processed['_id'].map(id2score)
final_df = processed.drop_duplicates('_id')

# ========= 保存每句句子得分 =========
sentence_score_dict = {}
split_dict = {}
for idx, row in processed.iterrows():
    sample_id = row['_id']
    sentence_score_dict.setdefault(sample_id, []).append(row['binary_entailment_prob'])
    split_dict[sample_id] = row['split']

sentence_score_records = []
caption_label_dict = processed.drop_duplicates('_id').set_index('_id')['caption_label'].to_dict()
for sample_id, scores in sentence_score_dict.items():
    sentence_score_records.append({
        "id": sample_id,
        "sentence_scores": scores,
        "caption_score": min(scores)
    })

with open(os.path.join(save_dir, "chartve_sentence_scores.jsonl"), "w") as f:
    for r in sentence_score_records:
        f.write(json.dumps(r) + "\n")

# ========= 全体指标 =========
sentence_level_labels = processed['sent_label'].tolist()
sentence_level_preds = (processed['binary_entailment_prob'] > 0.5).tolist()
sentence_acc = np.mean([p == l for p, l in zip(sentence_level_preds, sentence_level_labels)])

caption_preds = {k: min(v) > 0.5 for k, v in sentence_score_dict.items()}
caption_labels = {k: caption_label_dict[k] for k in caption_preds}
caption_acc = np.mean([caption_preds[k] == caption_labels[k] for k in caption_preds])
tau_all = kendalltau(
    list(caption_labels.values()),
    [min(sentence_score_dict[k]) for k in caption_preds],
    variant='c'
).statistic

# ========= 分 split 统计（只用 summarization / description） =========
splitwise_stats = {}
for split in ['summarization','description']:
    df_sub = processed[processed['split'] == split]
    sent_labels = df_sub['sent_label'].tolist()
    sent_preds = (df_sub['binary_entailment_prob'] > 0.5).tolist()
    sentence_acc_split = np.mean([p == l for p, l in zip(sent_preds, sent_labels)])

    sub_keys = [k for k in caption_preds if split_dict[k] == split]
    sub_labels = [caption_labels[k] for k in sub_keys]
    sub_preds = [caption_preds[k] for k in sub_keys]
    sub_scores = [min(sentence_score_dict[k]) for k in sub_keys]
    caption_acc_split = np.mean([p == l for p, l in zip(sub_preds, sub_labels)])
    tau_split = kendalltau(sub_labels, sub_scores, variant='c').statistic

    splitwise_stats[split] = {
        "sentence_level_accuracy": sentence_acc_split,
        "caption_level_accuracy": caption_acc_split,
        "kendalls_tau": tau_split
    }

# ========= 保存评估结果 =========
eval_summary = {
    "sentence_level_accuracy": sentence_acc,
    "caption_level_accuracy": caption_acc,
    "kendalls_tau": tau_all,
    "splitwise": splitwise_stats
}

with open(os.path.join(save_dir, "chartve_evaluation_summary.json"), "w") as f:
    json.dump(eval_summary, f, indent=2)

print("✅ 所有内容已保存:")
print(f"   - {os.path.join(save_dir, 'chartve_sentence_scores.jsonl')}")
print(f"   - {os.path.join(save_dir, 'chartve_evaluation_summary.json')}")

# ========= 输出摘要 =========
print(f"\n🔍 评估摘要:")
print(f"   句子级准确率: {sentence_acc:.4f}")
print(f"   段落级准确率: {caption_acc:.4f}")
print(f"   Kendall's Tau: {tau_all:.4f}")

print("\n📊 分类型统计:")
for split, stats in splitwise_stats.items():
    print(f"  {split}: 句子准确率={stats['sentence_level_accuracy']:.4f}, "
          f"段落准确率={stats['caption_level_accuracy']:.4f}, "
          f"Tau={stats['kendalls_tau']:.4f}")
