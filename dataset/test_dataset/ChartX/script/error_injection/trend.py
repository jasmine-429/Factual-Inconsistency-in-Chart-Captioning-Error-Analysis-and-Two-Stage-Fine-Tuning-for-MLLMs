import json
import re
from tqdm import tqdm
from collections import OrderedDict

# ===== 路径配置 =====
input_path = "/data/jguo376/project/dataset/test_dataset/ChartX/test_data/dataset/test_samples_id.json"
output_path = "/data/jguo376/project/dataset/test_dataset/ChartX/test_data/dataset/error_data/train_trend_errors.json"
log_path = "/data/jguo376/project/dataset/test_dataset/ChartX/test_data/dataset/error_data/error_log/trend_magnitude_error_doc.txt"

# ===== 构造反义词词典（可扩展）=====
ANTONYM_DICT = {
    # trend
    "increase": "decrease", "increases": "decreases", "increased": "decreased", "increasing": "decreasing",
    "rise": "fall", "rises": "falls", "rose": "fell", "rising": "falling",
    "grow": "decline", "grew": "declined", "growing": "declining",

    "climb": "drop", "climbs": "drops", "climbed": "dropped", "climbing": "dropping",
    "soar": "plummet", "soars": "plummets", "soared": "plummeted", "soaring": "plummeting",

    # magnitude
    "sharp": "slight", "sharply": "slightly",
    "dramatic": "modest", "dramatically": "modestly",
    "marked": "negligible", "markedly": "negligibly",
    "abrupt": "slow", "abruptly": "slowly",
    "substantial": "minimal", "substantially": "minimally",
    "intense": "faint", "intensely": "faintly",
}

TREND_WORDS = {
    "increase", "increases", "increased", "increasing",
    "rise", "rises", "rose", "rising",
    "grow", "grew", "growing", "decline", "declined", "declining",
    "climb", "climbs", "climbed", "climbing",
    "soar", "soars", "soared", "soaring",
}

MAGNITUDE_WORDS = {
    "sharp", "sharply", "dramatic", "dramatically",
    "marked", "markedly", "abrupt", "abruptly",
    "substantial", "substantially", "intense", "intensely",
}

# ===== 加载数据 =====
with open(input_path, "r") as f:
    data = json.load(f)

results = []
log_entries = []

for item in tqdm(data, desc="Injecting errors"):
    if item.get("label") != 1:
        continue

    sentence = item["sentence"]
    found_words = []

    for word, antonym in ANTONYM_DICT.items():
        pattern = r'\b' + re.escape(word) + r'\b'
        if re.search(pattern, sentence, flags=re.IGNORECASE):
            found_words.append((word, antonym))

    used_types = set()

    for word, antonym in found_words:
        if word in TREND_WORDS and "trend_error" in used_types:
            continue
        if word in MAGNITUDE_WORDS and "magnitude_error" in used_types:
            continue

        # 替换词，生成修改句子
        new_sentence = re.sub(r'\b' + re.escape(word) + r'\b', antonym, sentence, count=1, flags=re.IGNORECASE)
        error_type = "trend_error" if word in TREND_WORDS else "magnitude_error"
        used_types.add(error_type)

        # 构建新 entry，插入 error 字段，label = 0
        new_item = OrderedDict()
        for k in item:
            new_item[k] = item[k]
            if k == "imgname":
                new_item["id"] = item["id"]
        new_item["error"] = error_type
        new_item["sentence"] = new_sentence
        new_item["label"] = 0

        results.append(new_item)
        log_entries.append(f"{item['id']} | {error_type} | {word} → {antonym}")

# ===== 保存注入后的样本 =====
with open(output_path, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# ===== 保存注入日志 =====
with open(log_path, "w") as f:
    for entry in log_entries:
        f.write(entry + "\n")

print(f"✅ 错误注入完成，共生成 {len(results)} 条样本，保存至：{output_path}")
print(f"📝 注入词记录已保存至：{log_path}")
