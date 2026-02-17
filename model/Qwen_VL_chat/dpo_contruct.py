import json
import os

# ===== 路径配置 =====
# 含有 conversations/chosen/rejected 的“源”JSON（要被更新其 rejected.value）
source_file = "/data/jguo376/project/llama_factory/data/chart_caption_dpo_3error.json"

# 含有 generated_caption 的“目标”JSON（提供文本）
target_file = "/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/all_type/caption/deep_analysis/caption_all/qwen_caption_output.json"

# 输出文件（更新后的 source）
output_file = "/data/jguo376/project/llama_factory/data/chart_caption_dpo.json"


def normalize_path(p: str) -> str:
    """统一路径：去掉前缀'Chart/'、'./'，替换反斜杠为斜杠。"""
    if not p:
        return ""
    p = p.replace("\\", "/").strip()
    # 去掉开头的 ./ 或 // 等
    while p.startswith("./") or p.startswith("/"):
        p = p[1:]
    # 去掉 Chart/ 前缀
    if p.startswith("Chart/"):
        p = p[len("Chart/"):]
    return p

# ===== 读取数据 =====
with open(source_file, "r", encoding="utf-8") as f:
    source_data = json.load(f)

with open(target_file, "r", encoding="utf-8") as f:
    target_data = json.load(f)

# ===== 构建：图片路径 -> generated_caption 的映射 =====
gen_map = {}
dups = set()
for entry in target_data:
    img_path = entry.get("img")
    gen_cap = entry.get("generated_caption")
    key = normalize_path(img_path)
    if key:
        if key in gen_map:
            dups.add(key)
        if gen_cap:
            gen_map[key] = gen_cap

if dups:
    print(f"[Warn] Duplicate img keys in target: {len(dups)} (last one kept)")

# ===== 用 generated_caption 覆盖 source 的 rejected.value =====
updated, missing = 0, 0
for item in source_data:
    img_path = item.get("image")  # e.g. "Chart/bar_chart/png/bar_85.png"
    key = normalize_path(img_path)
    if not key:
        missing += 1
        continue

    gen_cap = gen_map.get(key)
    if gen_cap is None:
        # 目标里没找到对应图片
        missing += 1
        continue

    # 确保 rejected 字段存在
    if "rejected" not in item or not isinstance(item["rejected"], dict):
        item["rejected"] = {"from": "gpt", "value": ""}

    # 保留原值可选：item["rejected_original"] = item["rejected"].get("value")

    item["rejected"]["value"] = gen_cap
    # 若缺少 from 字段，补上
    item["rejected"].setdefault("from", "gpt")
    updated += 1

# ===== 保存结果 =====
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(source_data, f, ensure_ascii=False, indent=2)

print(f"完成！对齐并更新 {updated} 条；未匹配到的 {missing} 条。输出文件：{output_file}")
