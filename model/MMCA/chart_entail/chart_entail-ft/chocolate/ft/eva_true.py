import os
import torch
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from peft import PeftModel
from scipy.stats import kendalltau

# ===== 导入 MMCA 模型组件 =====
from mplug_owl.modeling_mplug_owl import MplugOwlForConditionalGeneration
from mplug_owl.processing_mplug_owl import MplugOwlProcessor

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.set_grad_enabled(False)

# ========= 配置 =========
model_path = "/data/jguo376/pretrained_models/mmca_base"   # ✅ 替换为 MMCA base 路径
lora_path = "/data/jguo376/project/model/MMCA/fine-tuning/output/sft_v0.1_ft_chartentail/checkpoint-5800"
use_lora = True
FACTUAL_ERROR_TYPES = ['label_error','magnitude_error','ooc_error','trend_error','value_error','nonsense_error']

# ========= 加载模型 =========
base_model = MplugOwlForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="cuda",
    trust_remote_code=True
).eval()

if use_lora:
    model = PeftModel.from_pretrained(base_model, lora_path).eval()
else:
    model = base_model

processor = MplugOwlProcessor.from_pretrained(model_path, trust_remote_code=True)

# ========= 数据 =========
with open("/data/jguo376/project/dataset/test_dataset/ChartX/test_data/dataset/combined_dataset.jsonl","r") as f:
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

# ========= 推理部分 (MMCA) =========
def get_prediction(processed_df):
    binary_positive_probs = []
    tokenizer = processor.tokenizer
    yes_id = tokenizer.convert_tokens_to_ids("yes")
    no_id = tokenizer.convert_tokens_to_ids("no")

    for row in tqdm(processed_df.itertuples(), total=len(processed_df)):
        image = Image.open(row.image_path).convert("RGB")

        # ✅ MMCA 的输入：直接 text + image
        inputs = processor(
            text=row.prompt,
            images=image,
            return_tensors="pt"
        ).to("cuda")

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits  # shape: [1, seq_len, vocab_size]
            next_token_logits = logits[0, -1]

        probs = torch.softmax(next_token_logits[[yes_id, no_id]], dim=0)
        entail_prob = probs[0].item()

        binary_positive_probs.append(entail_prob)

    processed_df['binary_entailment_prob'] = binary_positive_probs
    return processed_df

# ========= 处理数据 & 推理 =========
processed_error_analysis = proccess_samples(chocolate)
processed_error_analysis = get_prediction(processed_error_analysis)

_id_to_split = {sample["_id"]: sample["split"] for sample in chocolate}
processed_error_analysis["split"] = processed_error_analysis["_id"].map(_id_to_split)
id2score = processed_error_analysis.groupby('_id')['binary_entailment_prob'].min().to_dict()
processed_error_analysis["chartve_score"] = processed_error_analysis['_id'].map(id2score)

# ========= 评估 =========
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

with open("/data/jguo376/project/model/MMCA/chart_caption/chart_sft_entail/chartve_sentence_scores.jsonl", "w") as f:
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

# ========= 保存 =========
eval_summary = {
    "sentence_level_accuracy": sentence_acc,
    "caption_level_accuracy": caption_acc,
    "kendalls_tau": tau_all,
    "splitwise": splitwise_stats
}

os.makedirs("/data/jguo376/project/model/MMCA/chart_caption/chart_sft_entail", exist_ok=True)
with open("/data/jguo376/project/model/MMCA/chart_caption/chart_sft_entail/chartve_evaluation_summary.json", "w") as f:
    json.dump(eval_summary, f, indent=2)

print("\n✅ 所有内容已保存：")
print(" - chartve_sentence_scores.jsonl")
print(" - chartve_evaluation_summary.json")
