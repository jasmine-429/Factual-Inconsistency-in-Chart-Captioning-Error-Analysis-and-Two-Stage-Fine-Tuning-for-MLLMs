import os
import json
import re
import pandas as pd
from tqdm import tqdm
from io import StringIO
import random

# ========== 路径配置 ==========
annotation_path = "/data/jguo376/project/dataset/ChartX_dataset/ChartX/ChartX_annotation.json"
input_path = "/data/jguo376/project/dataset/test_dataset/ChartX/test_data/dataset/test_samples_id.json"
log_output_path = "/data/jguo376/project/dataset/test_dataset/ChartX/test_data/dataset/error_data/error_log/value_error_log.txt"
output_path = "/data/jguo376/project/dataset/test_dataset/ChartX/test_data/dataset/error_data/value_error_augmented.json"

# ===== 工具函数 =====
def extract_numbers(text):
    return re.findall(r'\d+(?:\.\d+)?', text)

def parse_csv(csv_str):
    clean = csv_str.replace("\\t", "\t").replace("\\n", "\n")
    return pd.read_csv(StringIO(clean), sep="\t")

def get_all_numeric_values(df):
    values = []
    for col in df.columns[1:]:
        for v in df[col]:
            try:
                values.append(float(v))
            except:
                continue
    return values

def find_cell(df, num_val):
    for row in df.index:
        for col in df.columns[1:]:
            try:
                val = float(df.loc[row, col])
                if abs(val - num_val) < 0.01:
                    return row, col
            except:
                continue
    return None, None

def get_row_candidates(df, row, exclude_val):
    vals = []
    for col in df.columns[1:]:
        try:
            v = float(df.loc[row, col])
            if abs(v - exclude_val) > 0.01:
                vals.append(v)
        except:
            continue
    return vals

def get_col_candidates(df, col, exclude_val):
    vals = []
    for row in df.index:
        try:
            v = float(df.loc[row, col])
            if abs(v - exclude_val) > 0.01:
                vals.append(v)
        except:
            continue
    return vals

# ===== 加载数据 =====
with open(annotation_path, 'r') as f:
    annotation_data = json.load(f)
img2csv = {item["img"]: item.get("csv", "") for item in annotation_data}

with open(input_path, 'r') as f:
    input_data = json.load(f)

augmented, logs = [], []

# ===== 主逻辑 =====
for item in tqdm(input_data):
    img_path = item["img"]
    sentence = item["sentence"]
    numbers = extract_numbers(sentence)
    csv_text = img2csv.get(img_path, "")
    if not csv_text:
        continue
    try:
        df = parse_csv(csv_text)
        all_values = get_all_numeric_values(df)
    except:
        continue

    for num_str in numbers:
        try:
            num_val = float(num_str)
        except:
            continue

        exact_match = any(abs(v - num_val) < 0.01 for v in all_values)
        fuzzy_match = not exact_match and any(abs(v - num_val) < 0.5 for v in all_values)

        if exact_match or fuzzy_match:
            row, col = find_cell(df, num_val)
            replacement = None
            strategy = "full"

            if col:
                col_candidates = get_col_candidates(df, col, num_val)
                if col_candidates:
                    replacement = random.choice(col_candidates)
                    strategy = "col"
            if replacement is None and row is not None:
                row_candidates = get_row_candidates(df, row, num_val)
                if row_candidates:
                    replacement = random.choice(row_candidates)
                    strategy = "row"
            if replacement is None:
                all_candidates = [v for v in all_values if abs(v - num_val) > 0.01]
                if all_candidates:
                    replacement = random.choice(all_candidates)
                    strategy = "full"

            if replacement is not None:
                replacement_clean = int(replacement) if replacement == int(replacement) else replacement
                sentence_new = sentence.replace(num_str, str(replacement_clean), 1)
                new_item = {
                    "chart_type": item["chart_type"],
                    "img": item["img"],
                    "imgname": item["imgname"],
                    "id": item["id"],
                    "source": item["source"],
                    "sentence": sentence_new,
                    "label": 0,
                    "error": "value_error"
                }
                augmented.append(new_item)
                logs.append(f'{item["id"]} | value_error | {num_str} → {replacement} | {strategy}')
                break  # 每句最多替一个

# ===== 保存输出 =====
with open(output_path, 'w') as f:
    json.dump(augmented, f, indent=2)

with open(log_output_path, 'w') as f:
    f.write("\n".join(logs))

print(f"✅ 已注入 {len(augmented)} 条 value_error")
print(f"📄 日志路径: {log_output_path}")
