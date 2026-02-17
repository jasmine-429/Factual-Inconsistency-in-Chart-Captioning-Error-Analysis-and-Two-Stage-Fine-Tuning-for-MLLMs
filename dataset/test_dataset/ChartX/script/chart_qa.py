#匹配QA
import json
from tqdm import tqdm

# ==== 路径配置 ====
input_sample_path = "/data/jguo376/project/dataset/test_dataset/ChartX/test_data/dataset/test_samples_extracted.json"
qa_sentence_path = "/data/jguo376/project/dataset/ChartX_dataset/ChartX/chartx_QA_llama_1sentence.json"
output_path = "/data/jguo376/project/dataset/test_dataset/ChartX/test_data/dataset/test_samples_with_QA.json"

# ==== 加载数据 ====
with open(input_sample_path, "r") as f:
    samples = json.load(f)

with open(qa_sentence_path, "r") as f:
    qa_sentences = json.load(f)

# ==== 构建 QA 映射（按 img 匹配）====
qa_map = {item["img"]: item["QA"] for item in qa_sentences}

# ==== 合并 QA_sentence 字段 ====
for item in tqdm(samples, desc="Merging QA_sentence"):
    item["QA_sentence"] = qa_map.get(item["img"], "")

# ==== 保存结果 ====
with open(output_path, "w") as f:
    json.dump(samples, f, indent=2)

print(f"✅ 已完成合并，共处理 {len(samples)} 条样本，结果保存至：{output_path}")
