from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import pandas as pd
import torch
import json
import numpy as np
from tqdm import tqdm
from scipy.stats import kendalltau
import os
from peft import PeftModel
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.set_grad_enabled(False)

lora_path = "/data/jguo376/project/llama_factory/src/saves/Qwen2-VL-7B-Instruct/lora/chart_caption_sft/checkpoint-870"
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


with open("/data/jguo376/project/dataset/test_dataset/ChartX/test_data/dataset/combined_dataset.jsonl", "r") as f:
    chocolate = [json.loads(line) for line in f]

def format_query(sentence):
    return f"Does the image entail this statement: \"{sentence}\""

def proccess_samples(samples):
    processed = []
    for sample in samples:
        caption_label = all(label not in FACTUAL_ERROR_TYPES for sent_labels in sample["labels"] for label in sent_labels)
        caption_label = int(caption_label)

        for sentence, sent_labels in zip(sample["generated_caption"], sample["labels"]):
            image_path = os.path.join("/data/jguo376/project/dataset/ChartX_dataset/ChartX", sample["img"].lstrip("./"))
            query = format_query(sentence)
            sent_label = 0 if any(l in FACTUAL_ERROR_TYPES for l in sent_labels) else 1
            row = [sample['_id'], image_path, query, sent_label, caption_label]
            processed.append(row)
    return pd.DataFrame(processed, columns=['_id','image_path','prompt','sent_label','caption_label'])

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

processed_error_analysis = proccess_samples(chocolate)
processed_error_analysis = get_prediction(processed_error_analysis)

_id_to_split = {sample["_id"]: sample["split"] for sample in chocolate}
processed_error_analysis["split"] = processed_error_analysis["_id"].map(_id_to_split)
id2score = processed_error_analysis.groupby('_id')['binary_entailment_prob'].min().to_dict()
processed_error_analysis["chartve_score"] = processed_error_analysis['_id'].map(id2score)

# Evaluate
final_df = processed_error_analysis.drop_duplicates('_id')
for split in ['summarization','description']:
    current_df = final_df.loc[final_df.split == split].dropna()
    tau = kendalltau(current_df.caption_label.values, current_df.chartve_score.values, variant='c').statistic
    print(f"Split {split}| Tau: {tau:.03f}")

# Sentence score & caption score
sentence_score_dict = {}
split_dict = {}
for idx, row in processed_error_analysis.iterrows():
    sample_id = row['_id']
    sentence_score_dict.setdefault(sample_id, []).append(row['binary_entailment_prob'])
    split_dict[sample_id] = row['split']

caption_label_dict = processed_error_analysis.drop_duplicates('_id').set_index('_id')['caption_label'].to_dict()
sentence_score_records = [{
    "id": k,
    "sentence_scores": v,
    "caption_score": min(v)
} for k, v in sentence_score_dict.items()]

with open("/data/jguo376/project/model/Qwen_VL_chat/chart_caption/chart_sft_entail/chartve_sentence_scores.jsonl", "w") as f:
    for r in sentence_score_records:
        f.write(json.dumps(r) + "\n")

sentence_level_labels = processed_error_analysis['sent_label'].tolist()
sentence_level_preds = (processed_error_analysis['binary_entailment_prob'] > 0.5).tolist()
sentence_acc = np.mean([p == l for p, l in zip(sentence_level_preds, sentence_level_labels)])

caption_preds = {k: min(v) > 0.5 for k, v in sentence_score_dict.items()}
caption_labels = {k: caption_label_dict[k] for k in caption_preds}
caption_acc = np.mean([caption_preds[k] == caption_labels[k] for k in caption_preds])
tau_all = kendalltau(
    list(caption_labels.values()),
    [min(sentence_score_dict[k]) for k in caption_preds],
    variant='c'
).statistic

splitwise_stats = {}
for split in ['summarization','description']:
    df_sub = processed_error_analysis[processed_error_analysis['split'] == split]
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

# Save
eval_summary = {
    "sentence_level_accuracy": sentence_acc,
    "caption_level_accuracy": caption_acc,
    "kendalls_tau": tau_all,
    "splitwise": splitwise_stats
}

with open("/data/jguo376/project/model/Qwen_VL_chat/chart_caption/chart_sft_entail/chartve_evaluation_summary.json", "w") as f:
    json.dump(eval_summary, f, indent=2)

print("\n✅ 所有内容已保存：")
print(" - chartve_sentence_scores.jsonl")
print(" - chartve_evaluation_summary.json")
