#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import shutil
from pathlib import Path
from tqdm import tqdm

# ====== 路径配置（改这里就行） ======
INPUT_JSON = "/data/jguo376/project/dataset/test_dataset/ChartX/train_data/data_6600/chart_entail_sharegpt.json"
IMAGE_ROOT = "/data/jguo376/project/dataset/ChartX_dataset/ChartX"
OUT_IMAGES_DIR = "/data/jguo376/project/mmca_images_all"
OUT_SHAREGPT_JSON = "/data/jguo376/project/model/MMCA/fine-tuning/dataset/train_chartentail/data/chart_entail_mmca_ready.json"
OUT_TRAIN_JSONL = "/data/jguo376/project/model/MMCA/fine-tuning/dataset/train_chartentail/data/train_entail_6600.jsonl"
TASK_TYPE = "sharegpt_chat_sft"

def derive_new_name(rel_path: str) -> str:
    """
    根据图像相对路径生成新图像文件名：
    - 如果原图像名是纯数字（如 493.png），则加 chart_type 前缀；
    - 如果图像名已包含非数字内容（如 bar_493.png），则保持不变。
    """
    p = Path(rel_path)
    parts = p.parts
    chart_type = parts[0] if len(parts) >= 2 else ""
    base_name = Path(parts[-1]).stem
    ext = Path(parts[-1]).suffix

    # 仅当文件名是纯数字时添加 chart_type 前缀
    if base_name.isdigit():
        return f"{chart_type}_{base_name}{ext}"
    else:
        return base_name + ext

def flatten_messages_to_text(messages) -> str:
    prompt = ("The following is a conversation between a curious human and AI assistant. "
              "The assistant gives helpful, detailed, and polite answers to the user's questions.\n")
    for m in messages:
        role = (m.get("role") or "").strip()
        content = (m.get("content") or "").strip()
        if role == "user":
            content = content.replace("<image>", "Human: <image>\nHuman: ").strip()
            if not content.startswith("Human:"):
                content = "Human: " + content
            prompt += content + "\n"
        elif role == "assistant":
            prompt += "AI: " + content + "\n"
    return prompt.strip()

def main():
    os.makedirs(OUT_IMAGES_DIR, exist_ok=True)
    out_images_path = Path(OUT_IMAGES_DIR)

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    updated_sharegpt = []
    seen_images = set()
    total, copied, missing, noimg = 0, 0, 0, 0

    for item in tqdm(data, desc="Copy & rename images"):
        total += 1
        images = item.get("images") or []
        if not images:
            noimg += 1
            continue

        rel_image = images[0].replace("Chart/", "")  # 去掉 Chart/ 前缀
        src_path = Path(IMAGE_ROOT) / rel_image

        if not src_path.exists():
            print(f"[WARNING] Image not found: {src_path}")
            missing += 1
            continue

        new_name = derive_new_name(rel_image)

        # ✅ 用重命名后的名称去重
        if new_name not in seen_images:
            dst_path = out_images_path / new_name
            shutil.copyfile(src_path, dst_path)
            seen_images.add(new_name)
            copied += 1

        # ✅ 更新 JSON 中的图像字段为重命名后路径
        new_item = dict(item)
        new_item["images"] = [new_name]
        updated_sharegpt.append(new_item)

    with open(OUT_SHAREGPT_JSON, "w", encoding="utf-8") as f:
        json.dump(updated_sharegpt, f, indent=2, ensure_ascii=False)

    with open(OUT_TRAIN_JSONL, "w", encoding="utf-8") as fout:
        for item in updated_sharegpt:
            image_file = item["images"][0]
            text = flatten_messages_to_text(item["messages"])
            sample = {
                "image": [image_file],
                "text": text,
                "task_type": TASK_TYPE
            }
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print("\n===== Summary =====")
    print(f"Total items:   {total}")
    print(f"Copied images: {copied}")
    print(f"Missing imgs:  {missing}")
    print(f"No-image samp: {noimg}")
    print(f"Images dir:    {OUT_IMAGES_DIR}")
    print(f"ShareGPT JSON: {OUT_SHAREGPT_JSON}")
    print(f"Train JSONL:   {OUT_TRAIN_JSONL}")
    print("✅ 下一步：训练配置中将 image_folder 指向上述 Images 目录，将 data_files 指向 JSONL 文件路径。")

if __name__ == "__main__":
    main()
