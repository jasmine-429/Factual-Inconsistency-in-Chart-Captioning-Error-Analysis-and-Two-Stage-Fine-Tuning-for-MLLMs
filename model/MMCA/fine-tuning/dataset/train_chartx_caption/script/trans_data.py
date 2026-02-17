import os
import re
import json
import random

# ====== 配置路径 ======
IN_PATH = "/data/jguo376/project/dataset/test_dataset/ChartX/chart_caption/SFT/data/chart_caption_sft.json"
OUT_PATH_5 = "/data/jguo376/project/model/MMCA/fine-tuning/dataset/train_chartx_caption/data/chartx_caption_sft_val.jsonl"
OUT_PATH_95 = "/data/jguo376/project/model/MMCA/fine-tuning/dataset/train_chartx_caption/data/chartx_caption_sft_train.jsonl"
IMG_BASE = "/data/jguo376/project/dataset/ChartX_dataset/ChartX"

SPLIT_RATIO = 0.05  # 5% 验证 / 测试

PREFIX = (
    "The following is a conversation between a curious human and AI assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

def guess_loader(path):
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(256)
        f.seek(0)
        if head.lstrip().startswith("["):
            return json.load(f)
        else:
            return [json.loads(line.strip()) for line in f if line.strip()]

def map_image_path(p, img_base):
    p = p.strip()
    if p.startswith("/"):
        return p
    m = re.match(r'(?i)^chart[\\/](.*)$', p)
    if m:
        rest = m.group(1).replace("\\", "/")
        return os.path.join(img_base, rest).replace("\\", "/")
    return os.path.join(img_base, p.replace("\\", "/")).replace("\\", "/")

def extract_user_and_assistant(messages):
    user_text = ""
    assistant_text = ""
    for m in messages or []:
        role = m.get("role", "")
        content = m.get("content", "")
        if role == "user" and not user_text:
            user_text = content
        elif role == "assistant" and not assistant_text:
            assistant_text = content
        if user_text and assistant_text:
            break
    return user_text, assistant_text

def build_text(user_content, assistant_content):
    cleaned = user_content.replace("<image>", "", 1).strip()
    return f"{PREFIX}\nHuman: <image>\nHuman: {cleaned}\nAI: {assistant_content or ''}"

def convert_split(in_path, out_path_5, out_path_95, img_base):
    data = guess_loader(in_path)
    total = len(data)
    print(f"总样本数：{total}")

    random.shuffle(data)
    split_idx = int(total * SPLIT_RATIO)

    val_set = data[:split_idx]
    train_set = data[split_idx:]

    def write_data(dataset, out_path):
        with open(out_path, "w", encoding="utf-8") as fout:
            for obj in dataset:
                user_content, assistant_content = extract_user_and_assistant(obj.get("messages", []))
                text = build_text(user_content, assistant_content)
                raw_images = obj.get("images", [])
                new_images = [map_image_path(p, img_base) for p in raw_images]
                new_obj = {
                    "image": new_images,
                    "text": text,
                    "task_type": "sharegpt_chat_sft"
                }
                fout.write(json.dumps(new_obj, ensure_ascii=False) + "\n")

    write_data(val_set, out_path_5)
    write_data(train_set, out_path_95)

    print(f"✅ 拆分完成：写出验证集 {len(val_set)} 条 → {out_path_5}")
    print(f"✅ 拆分完成：写出训练集 {len(train_set)} 条 → {out_path_95}")

if __name__ == "__main__":
    convert_split(IN_PATH, OUT_PATH_5, OUT_PATH_95, IMG_BASE)
