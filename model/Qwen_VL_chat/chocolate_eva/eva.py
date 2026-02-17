#作者给的评估代码
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import pandas as pd
import torch
import json
import numpy as np
from tqdm import tqdm
import requests
from scipy.stats import kendalltau
from transformers import AutoModelForVision2Seq, AutoProcessor
import os
from peft import PeftModel
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.set_grad_enabled(False)

lora_path = "/data/jguo376/project/llama_factory/src/saves/Qwen2-VL-7B-Instruct/lora/train_2025-08-04-00-40-35-part1/checkpoint-400"
FACTUAL_ERROR_TYPES = ['label_error', 'magnitude_error', 'ooc_error', 'trend_error','value_error','nonsense_error']
model_path = "/data/jguo376/pretrained_models/Qwen2.5-VL-7B-Instruct"
use_lora = True  # ✅ 是否加载 LoRA

# ✅ 加载 base 模型
base_model = AutoModelForVision2Seq.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="cuda",
    trust_remote_code=True
).eval()

# ✅ 判断是否使用 LoRA
if use_lora:
    model = PeftModel.from_pretrained(base_model, lora_path).eval()
else:
    model = base_model

# ✅ Processor 不变
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

with open("/data/jguo376/project/dataset/chcolate/CHOCOLATE/chocolate_test.jsonl", "r") as f:
    chocolate = [json.loads(line) for line in f]
def format_query(sentence):
    return f"Does the image entails this statement: \"{sentence}\"?"
def proccess_samples(samples): 
    processed = []
    for sample in samples:
        img_id = '-'.join(sample['_id'].split('-')[1:]).replace('pew_','').replace('vistext_','') 
        caption_label = all([ label not in FACTUAL_ERROR_TYPES for sent_labels in sample["labels"] for label in sent_labels]) 
        caption_label = int(caption_label)

        for sentence, sent_labels in zip(sample["sentences"], sample["labels"]):
            image_path = sample["local_image_path"]            
            query = format_query(sentence)
            sent_label = 0 if any([l in FACTUAL_ERROR_TYPES for l in sent_labels]) else 1
            #prompt =  "<chartqa>  " + query + " <s_answer>" 
            row = [sample['_id'], image_path, query, sent_label, caption_label]
            processed.append(row)
    processed = pd.DataFrame(processed, columns=['_id','image_path','prompt','sent_label','caption_label'])
    return processed
def get_prediction(processed_df):
    binary_positive_probs = []
    yes_id = processor.tokenizer.convert_tokens_to_ids("yes")
    no_id = processor.tokenizer.convert_tokens_to_ids("no")

    for row in tqdm(processed_df.itertuples(), total=len(processed_df)):
        image = Image.open(row.image_path).convert("RGB")

        # ✅ 构造消息
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": row.prompt},
                ],
            }
        ]

        # ✅ 构造 prompt，准备生成
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # ✅ 编码输入
        inputs = processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True
        ).to("cuda")

        # ✅ 前向传播，获取即将生成的第一个 token 的 logits
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits  # shape: [1, seq_len, vocab_size]
            next_token_logits = logits[0, -1]  # 最后一个位置是 assistant 回答第一个 token 的 logits

        # ✅ 计算 softmax 概率
        probs = torch.softmax(next_token_logits[[yes_id, no_id]], dim=0)
        entail_prob = probs[0].item()  # yes 的概率

        binary_positive_probs.append(entail_prob)

    processed_df['binary_entailment_prob'] = binary_positive_probs
    return processed_df

def get_split(sample_id):
    if "bard" in sample_id or "gpt4v" in sample_id:
        return "LVLM"
    elif "deplot" in sample_id:
        return "LLM"
    else:
        return "FT"
processed_chocolate = proccess_samples(chocolate)
processed_chocolate = get_prediction(processed_chocolate)    
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

with open("/data/jguo376/project/model/Qwen_VL_chat/chocolate_eva/part1-400/chartve_sentence_scores.jsonl", "w") as f:
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

with open("/data/jguo376/project/model/Qwen_VL_chat/chocolate_eva/part1-400/chartve_evaluation_summary.json", "w") as f:
    json.dump(eval_summary, f, indent=2)

print("✅ 所有内容已保存到：")
print("/data/jguo376/project/model/Qwen_VL_chat/chocolate_eva/org_eva/chartve_sentence_scores.jsonl")
print("/data/jguo376/project/model/Qwen_VL_chat/chocolate_eva/org_eva/chartve_evaluation_summary.json")
