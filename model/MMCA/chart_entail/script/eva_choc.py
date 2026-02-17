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
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
torch.set_grad_enabled(False)
DEVICE = "cuda"

from transformers import AutoTokenizer
from mplug_owl.modeling_mplug_owl import MplugOwlForConditionalGeneration
from mplug_owl.processing_mplug_owl import MplugOwlProcessor, MplugOwlImageProcessor
from peft import get_peft_model, LoraConfig

base_model_path = "/data/jguo376/pretrained_models/mplug-owl-llama-7b"
lora_path = "/data/jguo376/pretrained_models/MMCA/mmca_lora_weights.bin"

# ========= Processor / Tokenizer =========
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
image_processor = MplugOwlImageProcessor.from_pretrained(base_model_path)
processor = MplugOwlProcessor(image_processor=image_processor, tokenizer=tokenizer)

print("📦 加载 base 模型 + 注入 MMCA LoRA 权重")
base_model = MplugOwlForConditionalGeneration.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map="auto"
).eval()

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

# ========= 数据 & 标签 =========
FACTUAL_ERROR_TYPES = ['label_error', 'magnitude_error', 'ooc_error', 'trend_error', 'value_error', 'nonsense_error']

with open("/data/jguo376/project/dataset/chcolate/CHOCOLATE/chocolate_test.jsonl", "r") as f:
    chocolate = [json.loads(line) for line in f]

def format_query(sentence):
    return f'Does the image entails this statement: "{sentence}"?'

def proccess_samples(samples): 
    rows = []
    for sample in samples:
        caption_label = int(all([label not in FACTUAL_ERROR_TYPES for sent_labels in sample["labels"] for label in sent_labels]))
        for sentence, sent_labels in zip(sample["sentences"], sample["labels"]):
            rows.append([
                sample['_id'],
                sample["local_image_path"],
                format_query(sentence),
                0 if any([l in FACTUAL_ERROR_TYPES for l in sent_labels]) else 1,
                caption_label
            ])
    return pd.DataFrame(rows, columns=['_id','image_path','prompt','sent_label','caption_label'])

# ========= mask 工具 =========
def ensure_2d_mask(x, T, device=DEVICE, dtype=torch.long):
    """
    保证 mask 是 (1, T)，缺失->全1；过短->pad 1；过长->截断。
    """
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

# ========= 构建一次性输入（关键：传 labels，且 3 个外部 mask 裁成 T-1） =========
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

    input_ids = out["input_ids"]                     # (1, T)
    attention_mask_full = out.get("attention_mask")  # (1, T)
    pixel_values = out["pixel_values"]               # (1, C, H, W)

    T = input_ids.shape[1]
    tail = max(1, T - 1)   # 与内部 loss_mask 对齐

    # 外部 mask：裁成 T-1
    non_padding_mask = ensure_2d_mask(attention_mask_full, T)[:, :tail].contiguous()
    non_media_mask   = ensure_2d_mask(out.get("non_media_mask"), T)[:, :tail].contiguous()
    prompt_mask      = ensure_2d_mask(out.get("prompt_mask"), T)[:, :tail].contiguous()

    kwargs = dict(
        input_ids=input_ids,
        attention_mask=ensure_2d_mask(attention_mask_full, T),  # (1, T)
        pixel_values=pixel_values,
        num_images=torch.tensor([1], dtype=torch.long, device=DEVICE),

        # 这三个必须是 (1, T-1)
        non_padding_mask=non_padding_mask,
        non_media_mask=non_media_mask,
        prompt_mask=prompt_mask,

        # 关键新增：这版 forward 在 eval 也会用到 labels[:,1:]
        labels=input_ids.clone(),   # (1, T)
    )
    return kwargs, T, tail

# ========= 推理：取 yes 的 logit & softmax 概率 =========
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
            for k in ["input_ids","attention_mask","non_padding_mask","non_media_mask","prompt_mask"]:
                print(f"  {k}: {tuple(inputs[k].shape)}")
            first = False

        with torch.no_grad():
            out = model(**inputs)                # 不传 loss_mask
            next_token_logits = out.logits[0, -1]  # 第一个将生成的 token 的 logits

        # yes 的 logit
        yes_logit = next_token_logits[yes_id].item()
        # yes/no 的 softmax 概率（与你的 Qwen 评估一致）
        probs = torch.softmax(next_token_logits[[yes_id, no_id]], dim=0)
        yes_prob = probs[0].item()

        yes_logits.append(yes_logit)
        yes_probs.append(yes_prob)

    df["yes_logit"] = yes_logits
    df["binary_entailment_prob"] = yes_probs
    return df

def get_split(sample_id):
    if "bard" in sample_id or "gpt4v" in sample_id:
        return "LVLM"
    elif "deplot" in sample_id:
        return "LLM"
    else:
        return "FT"

# ========= 主流程 =========
processed_chocolate = proccess_samples(chocolate)
processed_chocolate = get_prediction(processed_chocolate)
processed_chocolate["split"] = processed_chocolate["_id"].apply(get_split)

# —— 下面评估/保存逻辑保持你的原样 ——
id2score = processed_chocolate.groupby('_id').binary_entailment_prob.min().to_dict()
processed_chocolate["chartve_score"] = processed_chocolate['_id'].map(id2score)
final_df = processed_chocolate.drop_duplicates('_id')

for split in ['LVLM','LLM','FT']:
    current_df = final_df.loc[final_df.split == split].dropna()
    tau = kendalltau(current_df.caption_label.values, current_df.chartve_score.values, variant='c').statistic
    print(f"Split {split}| Tau: {tau:.03f}")

sentence_score_dict = {}
split_dict = {}
for idx, row in processed_chocolate.iterrows():
    sample_id = row['_id']
    sentence_score_dict.setdefault(sample_id, []).append(row['binary_entailment_prob'])
    split_dict[sample_id] = row['split']

caption_label_dict = processed_chocolate.drop_duplicates('_id').set_index('_id')['caption_label'].to_dict()
sentence_score_records = [
    {"id": sid, "sentence_scores": sc, "caption_score": min(sc)}
    for sid, sc in sentence_score_dict.items()
]

save_dir = "/data/jguo376/project/model/MMCA/chart_entail/chart_entail-ft/chocolate/ft"
os.makedirs(save_dir, exist_ok=True)

with open(os.path.join(save_dir, "chartve_sentence_scores.jsonl"), "w") as f:
    for r in sentence_score_records:
        f.write(json.dumps(r) + "\n")

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

splitwise_stats = {}
for split in ['LVLM', 'LLM', 'FT']:
    df_sub = processed_chocolate[processed_chocolate['split'] == split]
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
