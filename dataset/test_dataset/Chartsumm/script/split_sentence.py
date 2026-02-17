import json
import spacy
from tqdm import tqdm

# ===== 初始化 spaCy 分句器 =====
nlp = spacy.load("en_core_web_sm")

# ===== 输入输出路径 =====
input_path = "/data/jguo376/project/dataset/chartsumm/train_s.json"  # <-- 你替换成真实路径
output_path = "/data/jguo376/project/dataset/test_dataset/Chartsumm/train_data/train_s_sentences.json"

# ===== 分句函数 =====
def spacy_split_sentences(text):
    if not text.strip():
        return []
    doc = nlp(text.strip())
    return [sent.text.strip() for sent in doc.sents]

# ===== 加载原始数据 =====
with open(input_path, "r") as f:
    data = json.load(f)

# ===== 提取句子并构建输出格式 =====
results = []
for item in tqdm(data, desc="Processing samples"):
    img_path = item["image"]
    imgname = img_path.split("/")[-1].split(".")[0]  # 去除后缀
    id_prefix = imgname
    sentence_id = 0

    for source_field in ["title", "summary"]:
        content = item.get(source_field, "")
        sentences = spacy_split_sentences(content)
        for sent in sentences:
            if sent:
                results.append({
                    "img": img_path,
                    "imgname": imgname,
                    "id": f"{id_prefix}_{sentence_id}",
                    "source": source_field,
                    "sentence": sent,
                    "label": 1
                })
                sentence_id += 1

# ===== 保存结果 =====
with open(output_path, "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"✅ 提取完成，共 {len(results)} 条句子，已保存至：{output_path}")
