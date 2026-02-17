import os
import json
import random
from collections import defaultdict
from tqdm import tqdm

# ===== 配置 =====
test_sent_path = "/data/jguo376/project/dataset/test_dataset/ChartX/test_data/dataset/test_samples_id.json"
annotation_path = "/data/jguo376/project/dataset/test_dataset/ChartX/test_data/dataset/test_samples_with_QA.json"
error_json_paths = {
    "trend_magnitude": "/data/jguo376/project/dataset/test_dataset/ChartX/test_data/dataset/error_data/trend_errors.json",
    "value_error": "/data/jguo376/project/dataset/test_dataset/ChartX/test_data/dataset/error_data/value_error_augmented.json",
    "label_error": "/data/jguo376/project/dataset/test_dataset/ChartX/test_data/dataset/error_data/label_error_augmented.json",
    "ooc_error": "/data/jguo376/project/dataset/test_dataset/ChartX/test_data/dataset/error_data/ooc_error_augmented.json",
    "nonsense_error": "/data/jguo376/project/dataset/test_dataset/ChartX/test_data/dataset/error_data/nonsence_error_augmented.json"
}
output_path = "/data/jguo376/project/dataset/test_dataset/ChartX/test_data/dataset/combined_chocolate_dataset.json"

POS_RATIO = 0.2
MAX_FULL_ERROR_RATIO = 0.1
random.seed(42)

# ===== 加载图元信息 =====
with open(annotation_path, "r") as f:
    annos = json.load(f)
imgfile2meta = {
    os.path.splitext(os.path.basename(x["img"]))[0]: x
    for x in annos
}

# ===== 加载原始测试句子 =====
with open(test_sent_path, "r") as f:
    test_data = json.load(f)

# ===== 加载错误句子 =====
error_maps = defaultdict(dict)

# trend + magnitude 特殊处理
trend_magnitude_path = error_json_paths.pop("trend_magnitude")
if os.path.exists(trend_magnitude_path):
    with open(trend_magnitude_path, "r") as f:
        for item in json.load(f):
            sid = item["id"]
            sid_base = sid[:-4] if sid.endswith("_ooc") else sid
            err_type = item.get("error")
            if err_type in {"trend_error", "magnitude_error"}:
                error_maps[sid_base][err_type] = item["sentence"]

# 加载其他类型错误
for err_type, path in error_json_paths.items():
    if not os.path.exists(path):
        continue
    with open(path, "r") as f:
        for item in json.load(f):
            sid = item["id"]
            sid_base = sid[:-4] if sid.endswith("_ooc") else sid
            error_maps[sid_base][err_type] = item["sentence"]

# ===== 只保留 summary 和 description，按 imgname + source 分组 =====
dataset = defaultdict(list)
for item in test_data:
    if item["source"] not in {"summarization", "description"}:
        continue
    key = f'{item["imgname"]}_{item["source"]}'
    dataset[key].append(item)

# ===== 构造正负样本 =====
final = []
full_error_count = 0
max_full_error = int(MAX_FULL_ERROR_RATIO * len(dataset))

for key, group in tqdm(dataset.items(), desc="Constructing"):
    imgname, source = key.rsplit("_", 1)
    example_img = group[0].get("img", "")
    img_base = os.path.splitext(os.path.basename(example_img))[0]
    meta = imgfile2meta.get(img_base)
    if not meta:
        continue

    group_sorted = sorted(group, key=lambda x: x["id"])
    sentences = [x["sentence"] for x in group_sorted]
    ids = [x["id"] for x in group_sorted]
    if not sentences or all(s.strip() == "" for s in sentences):
        continue

    is_positive = random.random() < POS_RATIO
    new_entry = {
        "chart_type": meta["chart_type"],
        "imgname": imgname,
        "img": meta["img"],
        "csv": meta.get("csv", ""),
        "title": meta.get("title", ""),
        "topic": meta.get("topic", ""),
        "split": source,
        "_id": f"{imgname}_{source}",
    }

    if is_positive:
        new_entry["generated_caption"] = sentences
        new_entry["labels"] = [[] for _ in sentences]
        new_entry["caption_consistent"] = "True"
    else:
        new_sents = sentences.copy()
        labels = [[] for _ in sentences]
        candidates = []

        for i, sid in enumerate(ids):
            sid_clean = sid[:-4] if sid.endswith("_ooc") else sid
            err_dict = error_maps.get(sid_clean, {})
            if not err_dict:
                continue

            # 优先 trend/magnitude，50%概率
            priority = [(i, s, t) for t, s in err_dict.items() if t in {"trend_error", "magnitude_error"}]
            others = [(i, s, t) for t, s in err_dict.items() if t not in {"trend_error", "magnitude_error"}]

            if priority and random.random() < 0.5:
                candidates.append(random.choice(priority))
            elif priority or others:
                candidates.append(random.choice(priority + others))

        if not candidates:
            continue

        n_total = len(sentences)
        # ✅ 全错条件判断：
        allow_full_error = (n_total == 1) or (full_error_count < max_full_error)
        max_errors = n_total if allow_full_error else n_total - 1
        if max_errors < 1:
            continue  # 跳过无法构造负样本的

        n_replace = random.randint(1, min(len(candidates), max_errors))
        sampled = random.sample(candidates, n_replace)

        for idx, new_sent, err_type in sampled:
            new_sents[idx] = new_sent
            labels[idx] = [err_type]

        if n_replace == n_total and n_total > 1:
            full_error_count += 1

        new_entry["generated_caption"] = new_sents
        new_entry["labels"] = labels
        new_entry["caption_consistent"] = "False"

    final.append(new_entry)

# ===== 保存输出 =====
with open(output_path, "w") as f:
    json.dump(final, f, indent=2)

print(f"✅ Saved {len(final)} samples to {output_path}")
print(f"🔢 Full-error negative samples (multi-sentence): {full_error_count}")
