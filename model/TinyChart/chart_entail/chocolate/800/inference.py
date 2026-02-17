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
from tinychart.conversation import conv_templates
from peft import PeftModel

# ========= 设置路径 =========
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

model_path = "/data/jguo376/pretrained_models/TinyChart-3B-768"
lora_path = "/data/jguo376/project/model/TinyChart/checkpoints/chart_entail/checkpoint-700"
use_lora = True

data_path = "/data/jguo376/project/dataset/chcolate/CHOCOLATE/chocolate_test.jsonl"
save_dir = "/data/jguo376/project/model/TinyChart/chart_entail/chocolate/800/org_yes_no"
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
model.eval()
print(f"Model loaded on device: {next(model.parameters()).device}")
print(f"Model dtype: {next(model.parameters()).dtype}")

# ========= Prompt =========
def format_query(sentence: str) -> str:
    return f'Does the image entail the following statement? "{sentence}" Output only one word: Yes or No.'

# ========= 数据预处理 =========
def proccess_samples(samples):
    rows = []
    for sample in samples:
        caption_label = int(all([label not in FACTUAL_ERROR_TYPES for sent_labels in sample["labels"] for label in sent_labels]))
        for sentence, sent_labels in zip(sample["sentences"], sample["labels"]):
            image_path = sample["local_image_path"]
            query = format_query(sentence)
            sent_label = 0 if any([l in FACTUAL_ERROR_TYPES for l in sent_labels]) else 1
            rows.append([sample['_id'], image_path, query, sent_label, caption_label])
    return pd.DataFrame(rows, columns=['_id','image_path','prompt','sent_label','caption_label'])

# ========= 解析 Yes/No =========
def parse_yes_no(text: str) -> int:
    if not text:
        return 0
    t = text.strip().lower()
    if t.startswith("yes"):
        return 1
    if t.startswith("no"):
        return 0
    for c in t:
        if c.isalpha():
            return 1 if c == "y" else 0
    return 0

# ========= 推理 =========
@torch.inference_mode()
def get_prediction_yesno(processed_df, tokenizer, model, image_processor, max_new_tokens=3):
    yesno_preds = []
    for row in tqdm(processed_df.itertuples(), total=len(processed_df)):
        try:
            image = Image.open(row.image_path).convert("RGB")
            image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0].unsqueeze(0)
            image_tensor = image_tensor.to("cuda").to(model.dtype)

            conv = conv_templates["phi"].copy()
            conv.append_message(conv.roles[0], row.prompt)
            conv.append_message(conv.roles[1], None)
            text_in = conv.get_prompt()

            inputs = tokenizer([text_in], return_tensors="pt").to("cuda")

            out_ids = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                images=image_tensor,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True
            )

            gen_ids = out_ids[0, inputs.input_ids.shape[1]:]
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            pred_bin = parse_yes_no(gen_text)
            yesno_preds.append(pred_bin)

        except Exception as e:
            print(f"[Error] row {row.Index}, id={row._id}: {e}")
            yesno_preds.append(0)

    processed_df["binary_entailment_pred"] = yesno_preds
    return processed_df

# ========= Split 类型 =========
def get_split(sample_id):
    if "bard" in sample_id or "gpt4v" in sample_id:
        return "LVLM"
    elif "deplot" in sample_id:
        return "LLM"
    else:
        return "FT"

# ========= 主流程 =========
print("Loading data...")
with open(data_path, "r") as f:
    chocolate = [json.loads(line) for line in f]

print("Processing samples...")
processed = proccess_samples(chocolate)
print("Running inference...")
processed = get_prediction_yesno(processed, tokenizer, model, image_processor)

processed["split"] = processed["_id"].apply(get_split)

# ========= 评估 =========
sentence_labels = processed['sent_label'].tolist()
sentence_preds  = processed['binary_entailment_pred'].tolist()
sentence_acc = np.mean([p == l for p, l in zip(sentence_preds, sentence_labels)])

sentence_pred_dict = {}
split_dict = {}
for idx, row in processed.iterrows():
    sid = row['_id']
    if sid not in sentence_pred_dict:
        sentence_pred_dict[sid] = []
        split_dict[sid] = row['split']
    sentence_pred_dict[sid].append(row['binary_entailment_pred'])

caption_preds = {k: int(min(v)) for k, v in sentence_pred_dict.items()}
caption_labels = processed.drop_duplicates('_id').set_index('_id')['caption_label'].to_dict()
caption_acc = np.mean([caption_preds[k] == caption_labels[k] for k in caption_preds])
tau_all = kendalltau(list(caption_labels.values()), [caption_preds[k] for k in caption_preds], variant='c').statistic

# ========= 按类别统计 =========
splitwise_stats = {}
for split in ['LVLM', 'LLM', 'FT']:
    df_sub = processed[processed['split'] == split]
    if len(df_sub) > 0:
        sent_acc = np.mean([
            p == l for p, l in zip(df_sub['binary_entailment_pred'], df_sub['sent_label'])
        ])
        sub_keys = [k for k in caption_preds if split_dict[k] == split]
        sub_labels = [caption_labels[k] for k in sub_keys]
        sub_preds  = [caption_preds[k] for k in sub_keys]
        cap_acc = np.mean([p == l for p, l in zip(sub_preds, sub_labels)])
        tau = kendalltau(sub_labels, sub_preds, variant='c').statistic if len(set(sub_labels)) > 1 else 0.0
    else:
        sent_acc = cap_acc = tau = 0.0

    splitwise_stats[split] = {
        "sentence_level_accuracy": sent_acc,
        "caption_level_accuracy": cap_acc,
        "kendalls_tau": tau,
        "num_samples": len(df_sub)
    }

# ========= 保存 =========
with open(os.path.join(save_dir, "chartve_evaluation_summary.json"), "w") as f:
    json.dump({
        "sentence_level_accuracy": sentence_acc,
        "caption_level_accuracy": caption_acc,
        "kendalls_tau": tau_all,
        "total_samples": len(processed),
        "total_captions": len(caption_labels),
        "splitwise": splitwise_stats
    }, f, indent=2)

print("评估完成 ✅")
print(f"句子级准确率: {sentence_acc:.4f}")
print(f"标题级准确率: {caption_acc:.4f}")
print(f"Kendall's Tau: {tau_all:.4f}")
