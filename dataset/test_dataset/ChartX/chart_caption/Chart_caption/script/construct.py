import json
import random
from collections import defaultdict
from copy import deepcopy
from tqdm import tqdm

# ===== 路径配置 =====
all_sentences_path = "/Users/gjx/Desktop/test_dataset/ChartX/train/train_samples_id.json"
output_path = "/Users/gjx/Desktop/test_dataset/ChartX/Chart_caption/data/dpo_augmented_small.json"

error_json_paths = {
    "trend_magnitude": "/Users/gjx/Desktop/test_dataset/ChartX/train/train_trend_errors.json",
    "value_error": "/Users/gjx/Desktop/test_dataset/ChartX/train/value_error_augmented.json",
    "label_error": "/Users/gjx/Desktop/test_dataset/ChartX/train/label_error_augmented.json",
    "ooc_error": "/Users/gjx/Desktop/test_dataset/ChartX/train/ooc_error_augmented.json",
    "nonsense_error": "/Users/gjx/Desktop/test_dataset/ChartX/train/nonsence_error_augmented.json"
}

# ===== 控制参数 =====
ERROR_TYPE_PROBS = {
    "trend_error": 0.2,
    "magnitude_error": 0.2,
    "value_error": 0.2,
    "label_error": 0.2,
    "ooc_error": 0.1,
    "nonsense_error": 0.1
}

NUM_ERROR_PROBS = {
    1: 0.6,
    2: 0.4
}

NEG_SAMPLE_COUNTS = {
    2: 0.7,
    1: 0.3
}

valid_sources = {"summarization", "description"}

# ===== 加载正样本句子 =====
with open(all_sentences_path, "r") as f:
    all_data = json.load(f)

filtered_data = [x for x in all_data if x["source"] in valid_sources]

# 按段落聚合
paragraphs = defaultdict(list)
for item in filtered_data:
    pid = (item["imgname"], item["source"])
    paragraphs[pid].append(item)

# ===== 加载错误句子 =====
error_dict = defaultdict(dict)
for err_type, path in error_json_paths.items():
    with open(path, "r") as f:
        for item in json.load(f):
            sid = item["id"]
            real_type = item.get("error", err_type)
            error_dict[real_type][sid] = item

# ===== 构造样本 =====
output = []

for (imgname, source), sents in tqdm(paragraphs.items()):
    sents_sorted = sorted(sents, key=lambda x: int(x["id"].split("_")[-1]))
    orig_texts = [x["sentence"] for x in sents_sorted]
    all_ids = [x["id"] for x in sents_sorted]
    para_id = f"{imgname}_{source}"

    # 正样本
    output.append({
        "id": para_id,
        "source": source,
        "imgname": imgname,
        "chart_type": sents_sorted[0]["chart_type"],
        "img": sents_sorted[0]["img"],
        "caption": " ".join(orig_texts),
        "caption_consistent": True,
        "labels": [[] for _ in orig_texts]
    })

    # 生成 2~3 个负样本，组合不重复
    generated_combinations = set()
    num_neg_samples = random.choices(list(NEG_SAMPLE_COUNTS.keys()), weights=NEG_SAMPLE_COUNTS.values())[0]

    tries = 0
    while len(generated_combinations) < num_neg_samples and tries < 10:
        tries += 1
        num_errors = random.choices(list(NUM_ERROR_PROBS.keys()), weights=NUM_ERROR_PROBS.values())[0]
        indices = random.sample(range(len(all_ids)), min(num_errors, len(all_ids)))
        indices = tuple(sorted(indices))

        if indices in generated_combinations:
            continue

        new_texts = deepcopy(orig_texts)
        labels = [[] for _ in new_texts]
        success = True

        for idx in indices:
            sent_id = all_ids[idx]
            possible_types = [etype for etype in error_dict if sent_id in error_dict[etype]]
            if not possible_types:
                success = False
                break

            weights = [ERROR_TYPE_PROBS.get(t, 0.1) for t in possible_types]
            err_type = random.choices(possible_types, weights=weights, k=1)[0]

            new_texts[idx] = error_dict[err_type][sent_id]["sentence"]
            labels[idx].append(err_type)

        if success:
            generated_combinations.add(indices)
            output.append({
                "id": para_id,
                "source": source,
                "imgname": imgname,
                "chart_type": sents_sorted[0]["chart_type"],
                "img": sents_sorted[0]["img"],
                "caption": " ".join(new_texts),
                "caption_consistent": False,
                "labels": labels
            })

# ===== 保存结果 =====
with open(output_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"✅ Done. Total samples: {len(output)}. Saved to: {output_path}")
