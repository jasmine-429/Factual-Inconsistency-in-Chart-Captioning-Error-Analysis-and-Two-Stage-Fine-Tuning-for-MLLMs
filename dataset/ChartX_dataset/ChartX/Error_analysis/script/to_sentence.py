# MMCA - 使用 spaCy 替代 nltk 进行分句处理
import json
import spacy

# 加载 spaCy 英文小模型（确保已执行过 python -m spacy download en_core_web_sm）
nlp = spacy.load("en_core_web_sm")

# 输入输出路径
input_path = "/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/sampled_exclude_5/unichart_caption_output.json"
output_path = "/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/sampled_exclude_5/sentence_dataset/unichart_sentence_output.json"

# 加载 JSON 数据
with open(input_path, "r", encoding="utf-8") as infile:
    data = json.load(infile)

# 分句处理
for item in data:
    caption = item.get("generated_caption", "")
    if isinstance(caption, str) and caption.strip():
        doc = nlp(caption)
        sentences = [sent.text.strip() for sent in doc.sents]
        item["generated_caption"] = sentences
    else:
        item["generated_caption"] = []

# 保存结果
with open(output_path, "w", encoding="utf-8") as outfile:
    json.dump(data, outfile, indent=2, ensure_ascii=False)

print(f"✅ 分句处理完成，共处理 {len(data)} 条样本，结果已保存到 {output_path}")
