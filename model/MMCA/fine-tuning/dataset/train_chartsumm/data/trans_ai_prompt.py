import json

input_file = "/data/jguo376/project/model/MMCA/fine-tuning/dataset/train_chartsumm/data/chartsumm_prompt_sft_val.jsonl"   # 原始文件
output_file = "/data/jguo376/project/model/MMCA/fine-tuning/dataset/train_chartsumm/data/chartsumm_vail_with_tags.jsonl"  # 输出文件

def add_tags(example):
    text = example["text"]

    if "short summary" in text:
        # 替换 AI: 为 AI: [SHORT]
        text = text.replace("AI: ", "AI: [SHORT] ", 1)
    elif "long summary" in text:
        # 替换 AI: 为 AI: [LONG]
        text = text.replace("AI: ", "AI: [LONG] ", 1)

    example["text"] = text
    return example

with open(input_file, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
    for line in f_in:
        if not line.strip():
            continue
        try:
            example = json.loads(line)
            example = add_tags(example)
            f_out.write(json.dumps(example, ensure_ascii=False) + "\n")
        except Exception as e:
            print("跳过错误行:", line[:100], e)

print(f"✅ 转换完成，新数据已保存到 {output_file}")
