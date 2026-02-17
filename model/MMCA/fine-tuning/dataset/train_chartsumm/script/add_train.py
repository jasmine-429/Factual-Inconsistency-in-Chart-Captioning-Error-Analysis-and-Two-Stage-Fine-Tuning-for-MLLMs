import os
import json
import random

# ===== 路径配置 =====
input_path = "/data/jguo376/project/dataset/chartsumm/test_s.json"   # 你的原始 JSON 文件路径
output_path = "/data/jguo376/project/model/MMCA/fine-tuning/dataset/train_chartsumm/data/output.jsonl"  # 转换后的 JSONL 文件
image_root = "/data/jguo376/project/dataset/chartsumm/chart_images"  # 图片根目录

# ===== 抽样比例 =====
SAMPLE_RATIO = 0.12

def convert_to_sharegpt(item):
    """
    将原始数据条目转成 sharegpt_chat_sft 格式
    """
    image_file = os.path.join(image_root, item["image"])  # 拼接完整路径
    summary = item.get("summary", "").strip()

    text_block = (
        "The following is a conversation between a curious human and AI assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
        "Human: <image>\n"
        "Human: Please generate a long summary of the chart.\n"
        f"AI: {summary}"
    )

    return {
        "image": [image_file],
        "text": text_block,
        "task_type": "sharegpt_chat_sft"
    }

def main():
    # 1. 加载数据
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 2. 随机抽样 8%
    sample_size = max(1, int(len(data) * SAMPLE_RATIO))
    sampled_data = random.sample(data, sample_size)

    # 3. 转换格式
    converted = [convert_to_sharegpt(item) for item in sampled_data]

    # 4. 保存为 JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in converted:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"✅ 共处理 {len(data)} 条，随机抽取 {len(sampled_data)} 条，保存至 {output_path}")

if __name__ == "__main__":
    main()
