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
from tinychart.eval.run_tiny_chart import inference_model

# ========= 设置路径 =========
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

model_path = "/data/jguo376/pretrained_models/TinyChart-3B-768"
lora_path = "/data/jguo376/project/model/TinyChart/checkpoints/chart_entail/checkpoint-800"  # 如无可为空
use_lora = False  # ✅ 是否加载 LoRA

data_path = "/data/jguo376/project/dataset/chcolate/CHOCOLATE/chocolate_test.jsonl"
save_dir = "/data/jguo376/project/model/TinyChart/chart_entail/chocolate/org"
os.makedirs(save_dir, exist_ok=True)

FACTUAL_ERROR_TYPES = ['label_error', 'magnitude_error', 'ooc_error', 'trend_error', 'value_error', 'nonsense_error']

# ========= 加载模型 =========
# ========= 加载模型（修复 lora_path 报错）=========
kwargs = {
    "model_base": None if not use_lora else model_path,
    "model_name": get_model_name_from_path(model_path),
    "device": "cuda:0"
}
if use_lora:
    kwargs["lora_path"] = lora_path

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path,
    **kwargs
)
model = model.float()

# ========= 数据预处理 =========
def format_query(sentence):
    return f"Does the image entails this statement: \"{sentence}\"?"

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

# ========= 适配 TinyChart 的推理函数 =========
def get_prediction(processed_df, tokenizer, model, image_processor):
    from tinychart.conversation import conv_templates

    binary_positive_probs = []
    yes_id = tokenizer.convert_tokens_to_ids("yes")
    no_id = tokenizer.convert_tokens_to_ids("no")

    for row in tqdm(processed_df.itertuples(), total=len(processed_df)):
        image = Image.open(row.image_path).convert("RGB")

        # 构造 prompt
        prompt = row.prompt
        conv = conv_templates["phi"].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        text = conv.get_prompt()

        # Tokenizer + image preprocess
        inputs = tokenizer([text], return_tensors="pt").to("cuda")
        image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0].unsqueeze(0).to("cuda")

        # Forward
        with torch.no_grad():
            outputs = model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                images=image_tensor,
                return_dict=True,
            )
            logits = outputs.logits
            next_token_logits = logits[0, -1]

        # Softmax over yes/no
        probs = torch.softmax(next_token_logits[[yes_id, no_id]], dim=0)
        entail_prob = probs[0].item()  # yes 的概率
        binary_positive_probs.append(entail_prob)

    processed_df["binary_entailment_prob"] = binary_positive_probs
    return processed_df


# ========= Split 类型解析 =========
def get_split(sample_id):
    if "bard" in sample_id or "gpt4v" in sample_id:
        return "LVLM"
    elif "deplot" in sample_id:
        return "LLM"
    else:
        return "FT"

# ========= 执行流程 =========
with open(data_path, "r") as f:
    chocolate = [json.loads(line) for line in f]

processed = proccess_samples(chocolate)
processed = get_prediction(processed, tokenizer, model, image_processor)
processed["split"] = processed["_id"].apply(get_split)
id2score = processed.groupby('_id').binary_entailment_prob.min().to_dict()
processed["chartve_score"] = processed['_id'].map(id2score)
final_df = processed.drop_duplicates('_id')

# ========= 保存每句句子得分 =========
sentence_score_dict = {}
split_dict = {}
for idx, row in processed.iterrows():
    sample_id = row['_id']
    if sample_id not in sentence_score_dict:
        sentence_score_dict[sample_id] = []
        split_dict[sample_id] = row['split']
    sentence_score_dict[sample_id].append(row['binary_entailment_prob'])

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

# ========= 评估指标 =========
sentence_level_labels = processed['sent_label'].tolist()
sentence_level_preds = (processed['binary_entailment_prob'] > 0.5).tolist()
sentence_acc = np.mean([p == l for p, l in zip(sentence_level_preds, sentence_level_labels)])

caption_preds = {k: min(v) > 0.5 for k, v in sentence_score_dict.items()}
caption_labels = {k: caption_label_dict[k] for k in caption_preds}
caption_acc = np.mean([caption_preds[k] == caption_labels[k] for k in caption_preds])
tau_all = kendalltau(list(caption_labels.values()), [min(sentence_score_dict[k]) for k in caption_preds], variant='c').statistic

# ========= 按类别统计 =========
splitwise_stats = {}
for split in ['LVLM', 'LLM', 'FT']:
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

print("✅ 所有内容已保存：")
print(os.path.join(save_dir, "chartve_sentence_scores.jsonl"))
print(os.path.join(save_dir, "chartve_evaluation_summary.json"))
