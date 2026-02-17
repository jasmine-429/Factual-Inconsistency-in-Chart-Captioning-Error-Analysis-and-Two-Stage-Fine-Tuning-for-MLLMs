#把42000的数据集转化一下，图片路径是原来的
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json

# ====== 把这三个路径按需改一下就能直接运行 ======
IN_PATH  = "/data/jguo376/project/dataset/test_dataset/ChartX/train_data/data_42000/chart_entail_sharegpt.json"     # 输入：旧格式（JSONL 或 JSON 数组）
OUT_PATH = "/data/jguo376/project/model/MMCA/fine-tuning/dataset/train_chartentail/data/mmca_42000.jsonl" # 输出：新格式 JSONL（单行一条）
IMG_BASE = "/data/jguo376/project/dataset/ChartX_dataset/ChartX"  # 图像根目录前缀

PREFIX = (
    "The following is a conversation between a curious human and AI assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

def guess_loader(path):
    """既支持 JSONL（每行一个对象），也支持单个 JSON 数组文件。"""
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(256)
        f.seek(0)
        if head.lstrip().startswith("["):  # JSON 数组
            data = json.load(f)
            for obj in data:
                yield obj
        else:  # JSONL
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

def map_image_path(p, img_base):
    """
    将 'Chart/.../file.png' → '/data/.../ChartX/.../file.png'
    若已是绝对路径，则保持不变；其它相对路径也拼到 img_base 下。
    """
    p = p.strip()
    if p.startswith("/"):
        return p
    m = re.match(r'(?i)^chart[\\/](.*)$', p)  # 忽略大小写匹配 'Chart/'
    if m:
        rest = m.group(1).replace("\\", "/")
        return os.path.join(img_base, rest).replace("\\", "/")
    return os.path.join(img_base, p.replace("\\", "/")).replace("\\", "/")

def extract_user_and_assistant(messages):
    """从 messages 里取第一条 user 和第一条 assistant 的 content。"""
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
    """构造最终 text：固定前缀 + Human: <image> + Human: 问句 + AI: 回答"""
    cleaned = user_content.replace("<image>", "", 1).strip()
    return f"{PREFIX}\nHuman: <image>\nHuman: {cleaned}\nAI: {assistant_content or ''}"

def convert(in_path, out_path, img_base):
    n_in, n_out = 0, 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for obj in guess_loader(in_path):
            n_in += 1

            # 取文本
            user_content, assistant_content = extract_user_and_assistant(obj.get("messages", []))
            text = build_text(user_content, assistant_content)

            # 处理图片路径（保留全部图片）
            raw_images = obj.get("images", [])
            new_images = [map_image_path(p, img_base) for p in raw_images]

            new_obj = {
                "image": new_images,
                "text": text,
                "task_type": "sharegpt_chat_sft"
            }
            fout.write(json.dumps(new_obj, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"✅ 转换完成：读取 {n_in} 条 → 写出 {n_out} 行到 {out_path}")

if __name__ == "__main__":
    convert(IN_PATH, OUT_PATH, IMG_BASE)
