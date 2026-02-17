import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from scipy.stats import kendalltau

# ========= 环境 & 路径 =========
sys.path.append("/data/jguo376/project/model/mPLUG-Owl/mPLUG-Owl")
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.set_grad_enabled(False)
DEVICE = "cuda"

from transformers import AutoTokenizer
from mplug_owl.modeling_mplug_owl import MplugOwlForConditionalGeneration
from mplug_owl.processing_mplug_owl import MplugOwlProcessor, MplugOwlImageProcessor
from peft import get_peft_model, LoraConfig, PeftModel

# ========= 模型路径 =========
model_path_for_weights = "/data/jguo376/pretrained_models/mmca_merged_model"
processor_ref_path = "/data/jguo376/pretrained_models/mplug-owl-llama-7b"
#lora_path = "/data/jguo376/project/model/MMCA/fine-tuning/output/sft_v0.1_ft_chartentail/checkpoint-5800"
lora_path = None  # 如果不想加载 LoRA

# ========= Processor / Tokenizer =========
tokenizer = AutoTokenizer.from_pretrained(processor_ref_path)
image_processor = MplugOwlImageProcessor.from_pretrained(processor_ref_path)
processor = MplugOwlProcessor(image_processor=image_processor, tokenizer=tokenizer)

print("📦 加载合并后模型权重（仅模型）")
base_model = MplugOwlForConditionalGeneration.from_pretrained(
    model_path_for_weights,
    torch_dtype=torch.float16,
    device_map=None   # 单卡更稳妥
).eval()
base_model.tie_weights()
base_model.to(DEVICE)

def attach_lora(base_model, lora_path):
    if lora_path is None:
        return base_model
    if os.path.isdir(lora_path):
        print(f"🔗 从目录加载 LoRA: {lora_path}")
        model = PeftModel.from_pretrained(base_model, lora_path)
        model.eval()
        return model
    print(f"🔗 从文件加载 LoRA: {lora_path}")
    peft_config = LoraConfig(
        target_modules=r'.*language_model.*\.(q_proj|v_proj)',
        inference_mode=True,
        r=8,
        lora_alpha=32,
        lora_dropout=0.05
    )
    model = get_peft_model(base_model, peft_config)
    state_dict = torch.load(lora_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

print("🧩 注入（可选）LoRA 权重")
model = attach_lora(base_model, lora_path)

# ========= 数据 =========
FACTUAL_ERROR_TYPES = ['label_error', 'magnitude_error', 'ooc_error',
                       'trend_error', 'value_error', 'nonsense_error']

with open("/data/jguo376/project/model/MMCA/chart_entail/chart_entail-ft/my_test/combined_dataset.jsonl", "r") as f:
    samples = [json.loads(line) for line in f]

def format_query(sentence):
    return f'Does the image entails this statement: "{sentence}"? Output only one word: Yes or No.'

def proccess_samples(samples): 
    rows = []
    for sample in samples:
        caption_label = int(all(
            label not in FACTUAL_ERROR_TYPES
            for sent_labels in sample["labels"]
            for label in sent_labels
        ))
        for sentence, sent_labels in zip(sample["generated_caption"], sample["labels"]):
            rows.append([
                sample['_id'],
                sample["img"],
                format_query(sentence),
                0 if any(l in FACTUAL_ERROR_TYPES for l in sent_labels) else 1,
                caption_label,
                sample.get("split", "unknown")
            ])
    return pd.DataFrame(rows, columns=['_id','image_path','prompt','sent_label','caption_label','split'])

# ========= mask 工具 =========
def ensure_2d_mask(x, T, device=DEVICE, dtype=torch.long):
    if x is None:
        x = torch.ones((1, T), dtype=dtype, device=device)
    elif isinstance(x, torch.Tensor):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.to(device).to(dtype)
        if x.shape[1] < T:
            pad = torch.ones((x.shape[0], T - x.shape[1]), dtype=dtype, device=device)
            x = torch.cat([x, pad], dim=1)
        elif x.shape[1] > T:
            x = x[:, :T]
    else:
        x = torch.ones((1, T), dtype=dtype, device=device)
    return x

# ========= 构建输入 =========
def build_inputs(image: Image.Image, prompt: str):
    pack = processor(text=[prompt], images=[image], return_tensors="pt")
    out = {}
    for k, v in pack.items():
        if isinstance(v, torch.Tensor):
            if v.dtype in (torch.float32, torch.float64):
                v = v.half()
            out[k] = v.to(DEVICE)
        else:
            out[k] = v
    input_ids = out["input_ids"]
    attention_mask_full = out.get("attention_mask")
    pixel_values = out["pixel_values"]
    T = input_ids.shape[1]
    tail = max(1, T - 1)
    non_padding_mask = ensure_2d_mask(attention_mask_full, T)[:, :tail].contiguous()
    non_media_mask   = ensure_2d_mask(out.get("non_media_mask"), T)[:, :tail].contiguous()
    prompt_mask      = ensure_2d_mask(out.get("prompt_mask"), T)[:, :tail].contiguous()
    kwargs = dict(
        input_ids=input_ids,
        attention_mask=ensure_2d_mask(attention_mask_full, T),
        pixel_values=pixel_values,
        num_images=torch.tensor([1], dtype=torch.long, device=DEVICE),
        non_padding_mask=non_padding_mask,
        non_media_mask=non_media_mask,
        prompt_mask=prompt_mask,
        labels=input_ids.clone(),
    )
    return kwargs, T, tail

# ========= 推理 =========
def get_prediction(df):
    yes_id = tokenizer.convert_tokens_to_ids("yes")
    no_id  = tokenizer.convert_tokens_to_ids("no")
    print(f"✅ yes_id={yes_id}, no_id={no_id}")
    yes_logits, yes_probs = [], []
    first = True
    for row in tqdm(df.itertuples(), total=len(df)):
        image = Image.open(row.image_path).convert("RGB")
        inputs, T, tail = build_inputs(image, row.prompt)
        if first:
            print(f"🔎 T={T}, T-1={tail}")
            first = False
        with torch.no_grad():
            out = model(**inputs)
            next_token_logits = out.logits[0, -1]
        yes_logit = next_token_logits[yes_id].item()
        probs = torch.softmax(next_token_logits[[yes_id, no_id]], dim=0)
        yes_prob = probs[0].item()
        yes_logits.append(yes_logit)
        yes_probs.append(yes_prob)
    df["yes_logit"] = yes_logits
    df["binary_entailment_prob"] = yes_probs
    return df

# ========= 主流程 =========
processed = proccess_samples(samples)
processed = get_prediction(processed)

id2score = processed.groupby('_id').binary_entailment_prob.min().to_dict()
processed["chartve_score"] = processed['_id'].map(id2score)
final_df = processed.drop_duplicates('_id')

# —— Split-wise Tau —— #
for split in ['summarization','description']:
    current_df = final_df.loc[final_df.split == split].dropna()
    if len(current_df) > 0:
        tau = kendalltau(current_df.caption_label.values, current_df.chartve_score.values, variant='c').statistic
        print(f"Split {split} | Tau: {tau:.03f}")

# ========= 保存 =========
sentence_score_dict = {}
split_dict = {}
for idx, row in processed.iterrows():
    sid = row['_id']
    sentence_score_dict.setdefault(sid, []).append(row['binary_entailment_prob'])
    split_dict[sid] = row['split']

caption_label_dict = processed.drop_duplicates('_id').set_index('_id')['caption_label'].to_dict()
sentence_score_records = [
    {"id": sid, "sentence_scores": sc, "caption_score": min(sc)}
    for sid, sc in sentence_score_dict.items()
]

save_dir = "/data/jguo376/project/model/MMCA/chart_entail/chart_entail-ft/my_test/org"
os.makedirs(save_dir, exist_ok=True)

with open(os.path.join(save_dir, "chartve_sentence_scores.jsonl"), "w") as f:
    for r in sentence_score_records:
        f.write(json.dumps(r) + "\n")

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

splitwise_stats = {}
for split in ['summarization','description']:
    df_sub = processed[processed['split'] == split]
    sent_preds = (df_sub['binary_entailment_prob'] > 0.5).tolist()
    sentence_acc_split = np.mean([p == l for p, l in zip(sent_preds, df_sub['sent_label'])])
    sub_keys = [k for k in caption_preds if split_dict[k] == split]
    sub_preds = [caption_preds[k] for k in sub_keys]
    sub_labels = [caption_labels[k] for k in sub_keys]
    sub_scores = [min(sentence_score_dict[k]) for k in sub_keys]
    caption_acc_split = np.mean([p == l for p, l in zip(sub_preds, sub_labels)])
    tau_split = kendalltau(sub_labels, sub_scores, variant='c').statistic
    splitwise_stats[split] = {
        "sentence_level_accuracy": sentence_acc_split,
        "caption_level_accuracy": caption_acc_split,
        "kendalls_tau": tau_split
    }

eval_summary = {
    "sentence_level_accuracy": sentence_acc,
    "caption_level_accuracy": caption_acc,
    "kendalls_tau": tau_all,
    "splitwise": splitwise_stats
}

with open(os.path.join(save_dir, "chartve_evaluation_summary.json"), "w") as f:
    json.dump(eval_summary, f, indent=2)

print("✅ 所有评估内容已保存到：")
print(os.path.join(save_dir, "chartve_sentence_scores.jsonl"))
print(os.path.join(save_dir, "chartve_evaluation_summary.json"))
