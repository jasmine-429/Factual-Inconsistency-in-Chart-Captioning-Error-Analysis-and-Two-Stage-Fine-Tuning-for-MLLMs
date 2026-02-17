import os
import json
import random

# ===== 配置 =====
input_files = [
    "/data/jguo376/project/dataset/chartsumm/train_k.json",
    "/data/jguo376/project/dataset/chartsumm/train_s.json"
]
output_json = "/data/jguo376/project/dataset/test_dataset/Chartsumm/caption_sft/train_data/data/train_k_s_sample5000_sharegpt.json" 
image_prefix = "chartsumm"   # 图片路径前缀
SAMPLE_SIZE = 4000
SEED = 42

random.seed(SEED)

def sample_and_convert(in_path, img_prefix, sample_size):
    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if len(data) <= sample_size:
        sampled = data
        print(f"⚠️ {os.path.basename(in_path)} 总数 {len(data)} ≤ {sample_size}，全部保留")
    else:
        sampled = random.sample(data, sample_size)
        print(f"✅ {os.path.basename(in_path)} 随机抽取 {sample_size} 条")

    out = []
    kept, skipped = 0, 0
    for it in sampled:
        img_name = str(it.get("image", "")).strip()
        summary = (it.get("summary") or "").strip()
        if not img_name or not summary:
            skipped += 1
            continue
        out.append({
            "messages": [
                {
                    "content": "<image>Please describe the chart.",
                    "role": "user"
                },
                {
                    "content": summary,
                    "role": "assistant"
                }
            ],
            "images": [
                f"{img_prefix}/{img_name}"
            ]
        })
        kept += 1

    print(f"📄 {os.path.basename(in_path)} | 保留 {kept} 条, 跳过 {skipped} 条")
    return out

if __name__ == "__main__":
    merged_data = []
    for in_path in input_files:
        merged_data.extend(sample_and_convert(in_path, image_prefix, SAMPLE_SIZE))

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)

    print(f"✅ 合并完成，总样本数: {len(merged_data)}")
    print(f"📄 输出文件: {output_json}")

