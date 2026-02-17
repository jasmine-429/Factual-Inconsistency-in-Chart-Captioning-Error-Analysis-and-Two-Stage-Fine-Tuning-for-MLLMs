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
use_lora = False  # 修改为 True

data_path = "/data/jguo376/project/dataset/chcolate/CHOCOLATE/chocolate_test.jsonl"
save_dir = "/data/jguo376/project/model/TinyChart/chart_entail/chocolate/800/org_prompt"
os.makedirs(save_dir, exist_ok=True)

FACTUAL_ERROR_TYPES = ['label_error', 'magnitude_error', 'ooc_error', 'trend_error', 'value_error', 'nonsense_error']

# ========= 加载模型 =========
print("Loading model...")
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path,
    model_base=None,  # 设为 None
    model_name=get_model_name_from_path(model_path),
    device="cuda:0"
)

if use_lora:
    print("Loading LoRA...")
    model = PeftModel.from_pretrained(model, lora_path)
    print("Merging LoRA weights...")
    model = model.merge_and_unload()
    print("LoRA weights loaded and merged.")

# 关键修复：统一数据类型
model = model.half()  # 而不是 model.float()
print(f"Model loaded on device: {next(model.parameters()).device}")
print(f"Model dtype: {next(model.parameters()).dtype}")

# ========= 数据预处理 =========
def format_query(sentence):
    return f"Does the image entails this statement: \"{sentence}\"? Answer with 'Yes' or 'No'."

def proccess_samples(samples):
    processed = []
    for sample in samples:
        img_id = '-'.join(sample['_id'].split('-')[1:]).replace('pew_','').replace('vistext_','') 
        caption_label = int(all([label not in FACTUAL_ERROR_TYPES for sent_labels in sample["labels"] for label in sent_labels]))
        for sentence, sent_labels in zip(sample["sentences"], sample["labels"]):
            image_path = sample["local_image_path"]
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
    
    print(f"Token IDs - yes: {yes_id}, no: {no_id}")

    for row in tqdm(processed_df.itertuples(), total=len(processed_df)):
        try:
            image = Image.open(row.image_path).convert("RGB")
            prompt = row.prompt
            conv = conv_templates["phi"].copy()
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            text = conv.get_prompt()

            inputs = tokenizer([text], return_tensors="pt", truncation=True, max_length=1024).to("cuda")
            # 关键修复：确保图像张量数据类型一致
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
            binary_positive_probs.append(0.5)  # 默认值

    processed_df["binary_entailment_prob"] = binary_positive_probs
    return processed_df

with open(data_path, "r") as f:
    chocolate = [json.loads(line) for line in f]

def get_split(sample_id):
    if "bard" in sample_id or "gpt4v" in sample_id:
        return "LVLM"
    elif "deplot" in sample_id:
        return "LLM"
    else:
        return "FT"
processed_chocolate = proccess_samples(chocolate)
processed_chocolate = get_prediction(processed_chocolate, tokenizer, model, image_processor)    
processed_chocolate["split"] = processed_chocolate["_id"].apply(get_split)
id2score = processed_chocolate.groupby('_id').binary_entailment_prob.min().to_dict()
processed_chocolate["chartve_score"] = processed_chocolate['_id'].map(id2score)
final_df = processed_chocolate.drop_duplicates('_id')
for split in ['LVLM','LLM','FT']:
    current_df = final_df.loc[final_df.split == split].dropna()
    tau = kendalltau(current_df.caption_label.values, current_df.chartve_score.values, variant='c').statistic
    print(f"Split {split}| Tau: {tau:.03f}")

# === 句子得分与 caption 分数记录 ===
sentence_score_dict = {}
split_dict = {}
for idx, row in processed_chocolate.iterrows():
    sample_id = row['_id']
    if sample_id not in sentence_score_dict:
        sentence_score_dict[sample_id] = []
        split_dict[sample_id] = row['split']
    sentence_score_dict[sample_id].append(row['binary_entailment_prob'])

sentence_score_records = []
caption_label_dict = processed_chocolate.drop_duplicates('_id').set_index('_id')['caption_label'].to_dict()

for sample_id, scores in sentence_score_dict.items():
    record = {
        "id": sample_id,
        "sentence_scores": scores,
        "caption_score": min(scores)
    }
    sentence_score_records.append(record)

with open(os.path.join(save_dir, "chartve_sentence_scores.jsonl"), "w") as f:
    for r in sentence_score_records:
        f.write(json.dumps(r) + "\n")

# === 全局评估指标 ===
sentence_level_labels = processed_chocolate['sent_label'].tolist()
sentence_level_preds = (processed_chocolate['binary_entailment_prob'] > 0.5).tolist()
sentence_acc = np.mean([p == l for p, l in zip(sentence_level_preds, sentence_level_labels)])

caption_preds = {k: min(v) > 0.5 for k, v in sentence_score_dict.items()}
caption_labels = {k: caption_label_dict[k] for k in caption_preds}
caption_acc = np.mean([caption_preds[k] == caption_labels[k] for k in caption_preds])
tau_all = kendalltau(
    list(caption_labels.values()),
    [min(sentence_score_dict[k]) for k in caption_preds],
    variant='c'
).statistic

# === 每类综合统计（sentence-level accuracy, caption-level accuracy, Kendall's Tau）===
splitwise_stats = {}
for split in ['LVLM', 'LLM', 'FT']:
    # Sentence-level accuracy
    df_sub = processed_chocolate[processed_chocolate['split'] == split]
    sent_labels = df_sub['sent_label'].tolist()
    sent_preds = (df_sub['binary_entailment_prob'] > 0.5).tolist()
    sentence_acc_split = np.mean([p == l for p, l in zip(sent_preds, sent_labels)])

    # Caption-level accuracy & tau
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

# === 保存评估结果 ===
eval_summary = {
    "sentence_level_accuracy": sentence_acc,
    "caption_level_accuracy": caption_acc,
    "kendalls_tau": tau_all,
    "splitwise": splitwise_stats  # 统一结构
}

with open(os.path.join(save_dir, "chartve_evaluation_summary.json"), "w") as f:
    json.dump(eval_summary, f, indent=2)

print("✅ 所有内容已保存到：")
print("/data/jguo376/project/model/Qwen_VL_chat/chocolate_eva/org_eva/chartve_sentence_scores.jsonl")
print("/data/jguo376/project/model/Qwen_VL_chat/chocolate_eva/org_eva/chartve_evaluation_summary.json")

