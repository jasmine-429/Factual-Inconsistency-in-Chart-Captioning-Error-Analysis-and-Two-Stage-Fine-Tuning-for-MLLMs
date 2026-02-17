
import os
import json
import pandas as pd
import spacy
import nltk
from nltk.stem import WordNetLemmatizer
from io import StringIO
from tqdm import tqdm

# ===== NLTK 资源路径 =====
nltk.data.path.append("/data/jguo376/conda_envs/mmca/nltk_data")
lemmatizer = WordNetLemmatizer()
nlp = spacy.load("en_core_web_sm")
# ====== 路径配置 ======
annotation_path = "/data/jguo376/project/dataset/ChartX_dataset/ChartX/ChartX_annotation.json"
input_path = "/data/jguo376/project/dataset/test_dataset/ChartX/test_data/dataset/test_samples_id.json"
output_path = "/data/jguo376/project/dataset/test_dataset/ChartX/test_data/dataset/error_data/label_error_augmented.json"
log_output_path = "/data/jguo376/project/dataset/test_dataset/ChartX/test_data/dataset/error_data/error_log/label_error_log.txt"


# ===== 工具函数 =====
def lemmatize(word):
    return lemmatizer.lemmatize(word.lower())

def normalize_img_key(img):
    return os.path.splitext(os.path.basename(img))[0]

def extract_csv_labels(csv_str):
    csv_str = csv_str.encode().decode("unicode_escape")  # 修复 \t \n
    df = pd.read_csv(StringIO(csv_str), sep="\t")
    row_labels = df.iloc[:, 0].astype(str).apply(str.strip).tolist()  # ← 加 strip
    col_labels = [c.strip() for c in df.columns.tolist()[1:]]         # ← 加 strip
    return row_labels, col_labels

def replace_entity_stable(sentence, row_labels, col_labels):
    doc = nlp(sentence)
    lemma_to_tokens = {}
    for token in doc:
        if token.pos_ not in {"NOUN", "PROPN"}:
            continue
        lemma = lemmatize(token.text)
        lemma_to_tokens.setdefault(lemma, []).append(token)

    row_lemma_map = {lemmatize(r): r for r in row_labels}
    col_lemma_map = {lemmatize(c): c for c in col_labels if c.lower() != "region"}

    print(f"\n[DEBUG] Sentence: {sentence}")
    print(f"  lemma_to_tokens: {lemma_to_tokens}")
    print(f"  row_lemmas: {list(row_lemma_map.keys())}")
    print(f"  col_lemmas: {list(col_lemma_map.keys())}")

    # === 尝试替换 row（North 等）
    for lemma, tokens in lemma_to_tokens.items():
        if lemma in row_lemma_map:
            current = row_lemma_map[lemma]
            candidates = [r for r in row_labels if r != current and r.lower() != "region"]
            if candidates:
                replacement = candidates[0]
                token = tokens[0]
                repl_token = replacement if token.text.islower() else replacement.capitalize()
                start, end = token.idx, token.idx + len(token.text)
                new_sent = sentence[:start] + repl_token + sentence[end:]
                return new_sent, token.text, repl_token, "row"

    # === 尝试替换 col（Doctors 等）
    for lemma, tokens in lemma_to_tokens.items():
        if lemma in col_lemma_map:
            current = col_lemma_map[lemma]
            candidates = [c for c in col_labels if c != current and c.lower() != "region"]
            if candidates:
                replacement = candidates[0]
                token = tokens[0]
                repl_token = replacement if token.text.islower() else replacement.capitalize()
                start, end = token.idx, token.idx + len(token.text)
                new_sent = sentence[:start] + repl_token + sentence[end:]
                return new_sent, token.text, repl_token, "col"

    return sentence, None, None, None

# ===== 加载数据 =====
with open(annotation_path, "r") as f:
    annotations = {normalize_img_key(item["img"]): item["csv"] for item in json.load(f)}

with open(input_path, "r") as f:
    samples = json.load(f)

output_data = []
log = []

# ===== 主处理循环 =====
for sample in tqdm(samples):
    if sample.get("source") == "title":
        continue

    img = normalize_img_key(sample["img"])
    sentence = sample["sentence"]

    if img not in annotations:
        print(f"[SKIP] img not found: {img}")
        continue

    csv_str = annotations[img]
    row_labels, col_labels = extract_csv_labels(csv_str)

    new_sent, orig_token, new_token, err_type = replace_entity_stable(sentence, row_labels, col_labels)

    if orig_token is not None:
        new_sample = sample.copy()
        new_sample["sentence"] = new_sent
        new_sample["error"] = "label_error"
        new_sample["label"] = 0
        output_data.append(new_sample)

        log_line = f'{sample["id"]} | label_error | {orig_token} → {new_token} | {err_type}'
        log.append(log_line)
    else:
        print(f"[INFO] No match found in: {sample['id']}")

# ===== 保存输出 =====
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w") as f:
    json.dump(output_data, f, indent=2)

with open(log_output_path, "w") as f:
    for line in log:
        f.write(line + "\n")
