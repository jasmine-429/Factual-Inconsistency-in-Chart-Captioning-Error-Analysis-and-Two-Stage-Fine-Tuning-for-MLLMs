#把数据集的段落分割为句子
import json
import spacy
from tqdm import tqdm

# ===== 加载 spaCy 英文模型 =====
nlp = spacy.load("en_core_web_sm")

# ===== 输入输出路径 =====
input_path = "/data/jguo376/project/dataset/test_dataset/ChartX/test_data/dataset/test_samples_with_QA.json"
output_path = "/data/jguo376/project/dataset/test_dataset/ChartX/test_data/dataset/test_sentences.json"

# ===== spaCy 分句函数 =====
def spacy_split_sentences(text):
    if not text.strip():
        return []
    doc = nlp(text.strip())
    return [sent.text.strip() for sent in doc.sents]

# ===== 加载原始数据 =====
with open(input_path, "r") as f:
    data = json.load(f)

# ===== 执行句子提取 =====
results = []
for item in tqdm(data, desc="Splitting with spaCy"):
    for field in ["title", "description", "summarization", "QA_sentence"]:
        content = item.get(field, "")
        for sent in spacy_split_sentences(content):
            if sent:
                results.append({
                    "chart_type": item["chart_type"],
                    "img": item["img"],
                    "imgname": item["imgname"],
                    "source": field,
                    "sentence": sent,
                    "label": 1
                })

# ===== 保存为标准 JSON 文件 =====
with open(output_path, "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"✅ spaCy 分句完成，共 {len(results)} 条句子，已保存为 JSON 文件：{output_path}")
