from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import pandas as pd
import torch
import json
import numpy as np
from tqdm import tqdm
from peft import PeftModel
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.set_grad_enabled(False)

# ===== 配置部分 =====
lora_path = "/data/jguo376/project/llama_factory/src/saves/Qwen2-VL-7B-Instruct/lora/train_2025-08-04-00-40-35-part1/checkpoint-400"
model_path = "/data/jguo376/pretrained_models/Qwen2.5-VL-7B-Instruct"
use_lora = True  # 是否加载 LoRA

# 输入输出路径
input_jsonl = "/data/jguo376/project/dataset/chartsumm/test_data.jsonl"  # 你的输入数据
image_base_dir = "/data/jguo376/project/dataset/chartsumm/chart_images"  # 图片文件夹
output_scores_jsonl = "/data/jguo376/project/model/Qwen_VL_chat/chocolate_eva/part1-400/chartsumm_sentence_scores.jsonl"
output_summary_json = "/data/jguo376/project/model/Qwen_VL_chat/chocolate_eva/part1-400/chartsumm_evaluation_summary.json"

# ===== 加载模型 =====
print("加载模型...")
base_model = AutoModelForVision2Seq.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="cuda",
    trust_remote_code=True
).eval()

if use_lora:
    model = PeftModel.from_pretrained(base_model, lora_path).eval()
    print("✅ LoRA 已加载")
else:
    model = base_model
    print("✅ 使用 base 模型")

processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

# ===== 读取数据 =====
print("读取数据...")
with open(input_jsonl, "r") as f:
    data = [json.loads(line) for line in f]

def format_query(sentence):
    return f"Does the image entails this statement: \"{sentence}\"?"

def process_samples(samples, image_base_dir):
    """处理新数据格式"""
    processed = []
    for sample in samples:
        sample_id = sample['_id']
        sentences = sample['sentences']
        
        # 构造图片路径
        image_path = os.path.join(image_base_dir, f"{sample_id}.png")
        
        # 检查图片是否存在
        if not os.path.exists(image_path):
            print(f"⚠️ 警告: 图片不存在 {image_path}")
            continue
        
        # 为每个句子创建一行
        for sentence in sentences:
            query = format_query(sentence)
            row = [sample_id, image_path, query, sentence]
            processed.append(row)
    
    processed_df = pd.DataFrame(processed, columns=['_id', 'image_path', 'prompt', 'sentence'])
    return processed_df

def get_prediction(processed_df):
    """对每个句子进行评分"""
    binary_positive_probs = []
    yes_id = processor.tokenizer.convert_tokens_to_ids("yes")
    no_id = processor.tokenizer.convert_tokens_to_ids("no")

    for row in tqdm(processed_df.itertuples(), total=len(processed_df)):
        try:
            image = Image.open(row.image_path).convert("RGB")

            # 构造消息
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": row.prompt},
                    ],
                }
            ]

            # 构造 prompt
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            # 编码输入
            inputs = processor(
                text=[text],
                images=[image],
                return_tensors="pt",
                padding=True
            ).to("cuda")

            # 前向传播
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                next_token_logits = logits[0, -1]

            # 计算 softmax 概率
            probs = torch.softmax(next_token_logits[[yes_id, no_id]], dim=0)
            entail_prob = probs[0].item()  # yes 的概率

            binary_positive_probs.append(entail_prob)
        
        except Exception as e:
            print(f"⚠️ 处理出错 {row._id}: {e}")
            binary_positive_probs.append(0.0)  # 出错时给默认值

    processed_df['binary_entailment_prob'] = binary_positive_probs
    return processed_df

# ===== 主流程 =====
print("处理样本...")
processed_df = process_samples(data, image_base_dir)

print(f"共有 {len(processed_df)} 个句子需要评分")
print("开始预测...")
processed_df = get_prediction(processed_df)

# ===== 按样本ID聚合结果 =====
print("聚合结果...")
sentence_score_dict = {}
for idx, row in processed_df.iterrows():
    sample_id = row['_id']
    if sample_id not in sentence_score_dict:
        sentence_score_dict[sample_id] = []
    sentence_score_dict[sample_id].append({
        'sentence': row['sentence'],
        'score': row['binary_entailment_prob']
    })

# ===== 保存句子级别得分 =====
sentence_score_records = []
for sample_id, sentence_info in sentence_score_dict.items():
    scores = [s['score'] for s in sentence_info]
    record = {
        "id": sample_id,
        "sentences": [s['sentence'] for s in sentence_info],
        "sentence_scores": scores,
        "caption_score": min(scores)  # caption得分 = 所有句子得分的最小值
    }
    sentence_score_records.append(record)

os.makedirs(os.path.dirname(output_scores_jsonl), exist_ok=True)
with open(output_scores_jsonl, "w") as f:
    for r in sentence_score_records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ===== 计算统计信息 =====
all_scores = processed_df['binary_entailment_prob'].tolist()
caption_scores = [min(sentence_score_dict[sid]) for sid in sentence_score_dict]

eval_summary = {
    "total_samples": len(sentence_score_dict),
    "total_sentences": len(processed_df),
    "avg_sentences_per_sample": len(processed_df) / len(sentence_score_dict),
    "sentence_level_stats": {
        "mean_score": float(np.mean(all_scores)),
        "median_score": float(np.median(all_scores)),
        "std_score": float(np.std(all_scores)),
        "min_score": float(np.min(all_scores)),
        "max_score": float(np.max(all_scores)),
        "positive_ratio": float(np.mean([s > 0.5 for s in all_scores]))
    },
    "caption_level_stats": {
        "mean_score": float(np.mean(caption_scores)),
        "median_score": float(np.median(caption_scores)),
        "std_score": float(np.std(caption_scores)),
        "min_score": float(np.min(caption_scores)),
        "max_score": float(np.max(caption_scores)),
        "positive_ratio": float(np.mean([s > 0.5 for s in caption_scores]))
    }
}

with open(output_summary_json, "w") as f:
    json.dump(eval_summary, f, indent=2, ensure_ascii=False)

print("\n✅ 评估完成！")
print(f"句子得分已保存到: {output_scores_jsonl}")
print(f"评估摘要已保存到: {output_summary_json}")
print(f"\n统计信息:")
print(f"  总样本数: {eval_summary['total_samples']}")
print(f"  总句子数: {eval_summary['total_sentences']}")
print(f"  平均每样本句子数: {eval_summary['avg_sentences_per_sample']:.2f}")
print(f"  句子级平均得分: {eval_summary['sentence_level_stats']['mean_score']:.4f}")
print(f"  Caption级平均得分: {eval_summary['caption_level_stats']['mean_score']:.4f}")