import sys
sys.path.append("/data/jguo376/project/model/TinyChart")

import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.stats import kendalltau

from tinychart.model.builder import load_pretrained_model
from tinychart.mm_utils import get_model_name_from_path
from tinychart.eval.run_tiny_chart import inference_model
from peft import PeftModel

# ========= 配置路径 =========
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

model_path = "/data/jguo376/pretrained_models/TinyChart-3B-768"
lora_path = "/data/jguo376/project/model/TinyChart/checkpoints/chart_entail/checkpoint-700"
use_lora = True

data_path = "/data/jguo376/project/dataset/chcolate/CHOCOLATE/chocolate_test.jsonl"
save_dir = "/data/jguo376/project/model/TinyChart/chart_entail/chocolate/800/700——yes"
os.makedirs(save_dir, exist_ok=True)

# ========= 加载模型 =========
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path),
    device="cuda:0"
)

if use_lora:
    print("Loading LoRA weights from:", lora_path)
    model = PeftModel.from_pretrained(model, lora_path)
    model = model.merge_and_unload()

model = model.half()

# ========= 工具函数 =========
FACTUAL_ERROR_TYPES = ['label_error', 'magnitude_error', 'ooc_error', 'trend_error', 'value_error', 'nonsense_error']

def parse_yesno(text):
    text = text.strip().lower()
    if text.startswith("yes") or text == "y":
        return 1
    if text.startswith("no") or text == "n":
        return 0
    return 0  # 默认无法判断为错误

def format_prompt(sentence):
    return f'Does the image entail the following statement? "{sentence}" Answer with Yes or No.'

# ========= 推理与评估 =========
with open(data_path, "r") as f:
    samples = [json.loads(line) for line in f]

sentence_score_dict = {}
caption_preds = {}
caption_labels = {}
split_dict = {}
all_results = []
sample_dict = {_s["_id"]: _s for _s in samples}

for sample in tqdm(samples):
    image_path = sample["local_image_path"]
    _id = sample["_id"]
    sentences = sample["sentences"]
    labels = sample["labels"]

    caption_label = int(all([label not in FACTUAL_ERROR_TYPES for label_list in labels for label in label_list]))
    caption_labels[_id] = caption_label

    sentence_preds = []

    for i, (sentence, sent_labels) in enumerate(zip(sentences, labels)):
        prompt = format_prompt(sentence)
        try:
            gen_text = inference_model(
                [image_path],
                prompt,
                model, tokenizer, image_processor, context_len,
                conv_mode="phi",
                max_new_tokens=10
            )[0]
        except Exception as e:
            print(f"[Generation Error] {_id}_{i}: {e}")
            gen_text = "[Generation Failed]"

        pred_label = parse_yesno(gen_text)
        true_label = 0 if any(l in FACTUAL_ERROR_TYPES for l in sent_labels) else 1

        sentence_id = f"{_id}_{i}"
        all_results.append({
            "_id": _id,
            "sentence_id": sentence_id,
            "sentence": sentence,
            "model_output": gen_text,
            "pred": pred_label,
            "label": true_label
        })

        sentence_preds.append(pred_label)

    sentence_score_dict[_id] = sentence_preds
    caption_preds[_id] = int(all(sentence_preds))
    split_dict[_id] = (
        "LVLM" if "bard" in _id or "gpt4v" in _id else
        "LLM" if "deplot" in _id else
        "FT"
    )

# ========= 全体评估 =========
sent_pred_all = [r["pred"] for r in all_results]
sent_label_all = [r["label"] for r in all_results]
sentence_acc = np.mean([p == l for p, l in zip(sent_pred_all, sent_label_all)])

caption_acc = np.mean([caption_preds[k] == caption_labels[k] for k in caption_labels])
caption_keys = list(caption_labels.keys())
tau_all = kendalltau(
    [caption_labels[k] for k in caption_keys],
    [min(sentence_score_dict[k]) for k in caption_keys],
    variant='c'
).statistic

# ========= 分 split 评估 =========
splitwise_stats = {}
for split in ['LVLM', 'LLM', 'FT']:
    sub_keys = [k for k in caption_keys if split_dict[k] == split]
    if not sub_keys:
        continue

    sent_preds = [p for k in sub_keys for p in sentence_score_dict[k]]
    sent_labels = []

    for k in [f"{cid}_{i}" for cid in sub_keys for i in range(len(sentence_score_dict[cid]))]:
        chart_id, sent_id = k.rsplit("_", 1)
        try:
            sent_label = 1 if all(
                l not in FACTUAL_ERROR_TYPES for l in sample_dict[chart_id]['labels'][int(sent_id)]
            ) else 0
        except:
            sent_label = 0
        sent_labels.append(sent_label)

    sentence_acc_split = np.mean([p == l for p, l in zip(sent_preds, sent_labels)])
    caption_acc_split = np.mean([caption_preds[k] == caption_labels[k] for k in sub_keys])
    tau_split = kendalltau(
        [caption_labels[k] for k in sub_keys],
        [min(sentence_score_dict[k]) for k in sub_keys],
        variant='c'
    ).statistic

    splitwise_stats[split] = {
        "sentence_level_accuracy": sentence_acc_split,
        "caption_level_accuracy": caption_acc_split,
        "kendalls_tau": tau_split
    }

# ========= 保存结果 =========
with open(os.path.join(save_dir, "chartve_sentence_scores.jsonl"), "w") as fout:
    for row in all_results:
        fout.write(json.dumps(row, ensure_ascii=False) + "\n")

eval_summary = {
    "sentence_level_accuracy": sentence_acc,
    "caption_level_accuracy": caption_acc,
    "kendalls_tau": tau_all,
    "splitwise": splitwise_stats
}
with open(os.path.join(save_dir, "chartve_evaluation_summary.json"), "w") as fout:
    json.dump(eval_summary, fout, indent=2)

print("✅ 推理 + 评估完成，保存路径：")
print("➡️", os.path.join(save_dir, "chartve_sentence_scores.jsonl"))
print("➡️", os.path.join(save_dir, "chartve_evaluation_summary.json"))
