import json
import random
import string
from tqdm import tqdm
from numpy.random import choice
import spacy
import torch
import math
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

# ===== 初始化 NLP 工具 =====
nlp = spacy.load("en_core_web_sm")

# ===== 初始化 GPT2 模型用于计算 PPL =====
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # ✅ 防止 padding 报错
model = GPT2LMHeadModel.from_pretrained("gpt2").cuda()
model.eval()

# ===== 计算句子 PPL（支持 batch）=====
def compute_ppl_batch(sent_list):
    encodings = tokenizer(sent_list, return_tensors="pt", padding=True, truncation=True)
    input_ids = encodings.input_ids.cuda()
    attention_mask = encodings.attention_mask.cuda()

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        ppl = torch.exp(loss)
        return ppl.item()  # 对 batch 求一个 loss（平均）

def is_high_ppl_batch_single(sentences, threshold=300.0):
    results = []
    for sent in tqdm(sentences, desc="Calculating PPL (per sentence)"):
        encodings = tokenizer(sent, return_tensors="pt", truncation=True)
        input_ids = encodings.input_ids.cuda()
        attention_mask = encodings.attention_mask.cuda()

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            ppl = torch.exp(loss).item()

        results.append(ppl > threshold)
    return results

# ===== 强化垃圾 token 生成器 =====
def generate_garbage_token():
    garbage_types = ["symbol", "rand_letters", "digit_sym", "repeat_char", "alphanum_mix"]
    gtype = random.choice(garbage_types)
    if gtype == "symbol":
        return random.choice(["@#$%", "&*()", "{}[]", "<>?", "|\\~"])
    elif gtype == "rand_letters":
        return ''.join(random.choices(string.ascii_lowercase, k=random.randint(2, 5)))
    elif gtype == "digit_sym":
        return ''.join(random.choices("0123456789", k=3)) + ''.join(random.choices("@#$%^&*", k=2))
    elif gtype == "repeat_char":
        c = random.choice(string.ascii_lowercase)
        return c * random.randint(2, 6)
    elif gtype == "alphanum_mix":
        return ''.join([random.choice(string.ascii_lowercase) + random.choice("0123456789") for _ in range(3)])

# ===== 扰动方法 =====
def shuffle_words(sentence, ratio=0.4):
    words = sentence.split()
    n = max(1, int(len(words) * ratio))
    indices = random.sample(range(len(words)), n)
    to_shuffle = [words[i] for i in indices]
    random.shuffle(to_shuffle)
    for idx, val in zip(indices, to_shuffle):
        words[idx] = val
    return " ".join(words)

def break_grammar_structure(sentence, ratio=0.4):
    doc = nlp(sentence)
    tokens = list(doc)
    n = max(1, int(len(tokens) * ratio))
    selected = random.sample(tokens, n)
    selected_text = [t.text for t in selected]
    rest = [t.text for t in tokens if t not in selected]
    random.shuffle(selected_text)
    return " ".join(selected_text + rest)

def insert_garbage(sentence, ratio=0.4):
    words = sentence.split()
    n_insert = max(1, int(len(words) * ratio))
    for _ in range(n_insert):
        idx = random.randint(0, len(words))
        words.insert(idx, generate_garbage_token())
    return " ".join(words)

def drop_subject_or_verb(sentence):
    doc = nlp(sentence)
    tokens = [t.text for t in doc]
    for i, token in enumerate(doc):
        if token.dep_ in {"nsubj", "ROOT"}:
            del tokens[i]
            return " ".join(tokens)
    for i, token in enumerate(doc):
        if token.pos_ in {"NOUN", "VERB"}:
            del tokens[i]
            break
    return " ".join(tokens)

perturb_fns = {
    "shuffle_words": shuffle_words,
    "break_grammar_structure": break_grammar_structure,
    "insert_garbage": insert_garbage,
    "drop_subject_or_verb": drop_subject_or_verb,
}

perturb_weights = {
    "shuffle_words": 0.25,
    "break_grammar_structure": 0.30,
    "insert_garbage": 0.25,
    "drop_subject_or_verb": 0.20
}

# ===== 主函数 =====
def perturb_json_file(input_path, output_json_path, log_txt_path="log.txt", ppl_threshold=300.0):
    with open(input_path, "r") as f:
        data = json.load(f)

    method_stat = {k: 0 for k in perturb_fns}
    total_perturbed = 0
    output = []
    log_lines = []

    orig_sents, pert_sents, indices, method_names = [], [], [], []

    for idx, item in enumerate(data):
        if item.get("source") == "title":   # ❌ 不再保留 title
            continue
        if len(item.get("sentence", "").split()) < 10:  # ❌ 不再保留短句
            continue

        orig_sent = item["sentence"]
        method_name = choice(list(perturb_weights.keys()), p=list(perturb_weights.values()))
        fn = perturb_fns[method_name]
        perturbed = fn(orig_sent)

        orig_sents.append(orig_sent)
        pert_sents.append(perturbed)
        indices.append(idx)
        method_names.append(method_name)

    keep_flags = is_high_ppl_batch_single(pert_sents, threshold=ppl_threshold)

    for i, keep in enumerate(keep_flags):
        item = data[indices[i]]
        if keep:
            item["sentence"] = pert_sents[i]
            item["label"] = 0
            item["error"] = "nonsense_error"
            item["method"] = method_names[i]
            method_stat[method_names[i]] += 1
            total_perturbed += 1

            log_lines.append(f"ID: {item.get('id', '')}\nMethod: {method_names[i]}\nOriginal: {orig_sents[i]}\nPerturbed: {pert_sents[i]}\n---")

            output.append(item)

    with open(output_json_path, "w") as f:
        json.dump(output, f, indent=2)
    with open(log_txt_path, "w") as f:
        f.write("\n".join(log_lines))

    print(f"\n✅ 总共扰动成功 {total_perturbed} 条样本")
    print(f"📄 日志写入完成：{log_txt_path}")
    print("🔧 各方法注入统计：")
    for k, v in method_stat.items():
        print(f" - {k}: {v}")

input_path = "/data/jguo376/project/dataset/test_dataset/ChartX/test_data/dataset/test_samples_id.json"  # 原始图表数据路径

# ===== 使用示例 =====
perturb_json_file(
    "/data/jguo376/project/dataset/test_dataset/ChartX/test_data/dataset/test_samples_id.json",
    "/data/jguo376/project/dataset/test_dataset/ChartX/test_data/dataset/error_data/nonsence_error_augmented.json",
    "/data/jguo376/project/dataset/test_dataset/ChartX/test_data/dataset/error_data/error_log/nonsence_log.txt",
    ppl_threshold=300.0
)
