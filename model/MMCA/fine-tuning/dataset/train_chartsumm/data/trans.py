import json

# === 输入输出路径 ===
input_jsonl = "/data/jguo376/project/model/MMCA/fine-tuning/dataset/train_chartsumm/data/chartsumm_caption_sft_val.jsonl"
output_jsonl = "/data/jguo376/project/model/MMCA/fine-tuning/dataset/train_chartsumm/data/chartsumm_prompt_sft_val.jsonl"

# === 指令替换逻辑 ===
def replace_instruction(data):
    image_path = data["image"][0]
    
    if "train_k" in image_path:
        new_instruction = "Please generate a short summary of the chart."
    elif "train_s" in image_path:
        new_instruction = "Please generate a long summary of the chart."
    else:
        return data  # 不替换
    
    # 替换 "Human: Please describe the chart." 那一行
    text_lines = data["text"].split("\n")
    for i, line in enumerate(text_lines):
        if line.strip() == "Human: Please describe the chart.":
            text_lines[i] = f"Human: {new_instruction}"
            break
    data["text"] = "\n".join(text_lines)
    return data

# === 主程序 ===
with open(input_jsonl, "r", encoding="utf-8") as f_in, open(output_jsonl, "w", encoding="utf-8") as f_out:
    for line in f_in:
        item = json.loads(line)
        updated_item = replace_instruction(item)
        f_out.write(json.dumps(updated_item, ensure_ascii=False) + "\n")

print("✅ 完成指令替换并保存到:", output_jsonl)
