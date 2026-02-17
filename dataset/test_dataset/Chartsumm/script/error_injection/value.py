import json
import re
import random
from tqdm import tqdm
from pathlib import Path

# ===== 路径配置 =====
input_path = "/data/jguo376/project/dataset/test_dataset/Chartsumm/train_data/train_s/train_s_sentences.json"
raw_data_path = "/data/jguo376/project/dataset/chartsumm/train_s.json"
output_path = "/data/jguo376/project/dataset/test_dataset/Chartsumm/train_data/train_s/value_error_augmented.json"
log_output_path = "/data/jguo376/project/dataset/test_dataset/Chartsumm/train_data/train_s/error_log/value_error_log.txt"

# ===== 数值提取（支持 , 和 %）=====
def extract_numbers(text):
    pattern = r'\d{1,3}(?:,\d{3})*(?:\.\d+)?%?|\d+(?:\.\d+)?%?'
    return re.findall(pattern, text)

# ===== 获取格式 & 精度（是否有 , 和 %）=====
def match_format(original_str, value):
    has_comma = ',' in original_str
    has_percent = '%' in original_str
    decimal_len = len(original_str.split('.')[-1].replace('%','')) if '.' in original_str else 0
    fmt = f"{{:,.{decimal_len}f}}" if has_comma else f"{{:.{decimal_len}f}}"
    formatted = fmt.format(value)
    return formatted + '%' if has_percent else formatted, decimal_len

# ===== 过滤候选：显示不同 + 值不同 =====
def filter_candidates(candidates, original_val, original_str):
    original_fmt, decimal_len = match_format(original_str, original_val)
    def is_display_different(v):
        v_fmt, _ = match_format(original_str, v)
        return v_fmt != original_fmt
    return [v for v in candidates if abs(v - original_val) > 0.5 and is_display_different(v)]

# ===== 加载数据 =====
with open(input_path, "r") as f:
    sentence_data = json.load(f)
with open(raw_data_path, "r") as f:
    chart_data = json.load(f)

# 构建 image → raw_data 映射
img2data = {item["image"]: item for item in chart_data}

augmented = []
logs = []

# ===== 主逻辑 =====
for item in tqdm(sentence_data, desc="Injecting value_error"):
    sentence = item["sentence"]
    img = item["img"]
    imgname = item["imgname"]
    img_id = item["id"]
    source = item["source"]
    if source == "title":
        continue  # ✅ 跳过标题类句子

    raw = img2data.get(img, None)
    if not raw or "data" not in raw:
        continue

    data_dict = raw["data"]
    numbers = extract_numbers(sentence)
    matched = False

    for num_str in numbers:
        clean_str = num_str.replace(',', '').replace('%', '')
        try:
            val = float(clean_str)
        except:
            continue

        # 尝试模糊匹配属于哪个字段
        matched_field = None
        for key, series in data_dict.items():
            for v in series:
                try:
                    if abs(round(float(v), len(clean_str.split('.')[-1]) if '.' in clean_str else 0) - round(val, len(clean_str.split('.')[-1]) if '.' in clean_str else 0)) < 0.5:
                        matched_field = key
                        break
                except:
                    continue
            if matched_field:
                break

        if not matched_field:
            continue

        # 在该字段中挑选候选值
        try:
            field_vals = [float(v) for v in data_dict[matched_field]]
        except:
            continue
        candidates = filter_candidates(field_vals, val, num_str)
        if not candidates:
            continue

        replacement = random.choice(candidates)
        replacement_formatted, _ = match_format(num_str, replacement)
        sentence_new = sentence.replace(num_str, replacement_formatted, 1)

        new_item = {
            "img": img,
            "imgname": imgname,
            "id": img_id,
            "source": source,
            "sentence": sentence_new,
            "label": 0,
            "error": "value_error"
        }
        augmented.append(new_item)
        logs.append(f"{img_id} | value_error | {num_str} → {replacement_formatted} | field={matched_field}")
        matched = True
        break  # 每句只替换一个

# ===== 保存输出 =====
Path(log_output_path).parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w") as f:
    json.dump(augmented, f, indent=2)
with open(log_output_path, "w") as f:
    f.write("\n".join(logs))

print(f"✅ 成功注入 value_error 共 {len(augmented)} 条")
print(f"📄 日志保存至: {log_output_path}")
