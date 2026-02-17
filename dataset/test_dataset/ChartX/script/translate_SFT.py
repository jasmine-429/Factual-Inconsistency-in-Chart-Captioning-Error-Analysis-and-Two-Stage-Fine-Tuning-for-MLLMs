#转换为sft格式，但不是最终格式
import json

# ===== 输入输出路径 =====
input_file = "/data/jguo376/project/dataset/test_dataset/ChartX/test_data/dataset/test_balanced_mixed.json"
output_file = "/data/jguo376/project/dataset/test_dataset/ChartX/test_data/dataset/test_sft.json"

# ===== 后缀映射规则 =====
def get_suffix(item):
    if item["label"] == 1:
        return "pos"
    err = item.get("error")
    if err == "value_error":
        return "val"
    elif err == "label_error":
        return "lab"
    elif err == "trend_error":
        return "trend"
    elif err == "magnitude_error":
        return "mag"
    elif err == "ooc_error":
        return "ooc"
    elif err == "nonsense_error":
        return "non"
    else:
        return "unk"

# ===== 加载样本 =====
with open(input_file) as f:
    data = json.load(f)

# ===== 转换为 SFT 格式，避免 ID 重复拼接 =====
sft_data = []
for item in data:
    suffix = get_suffix(item)
    # 如果 id 已经有后缀，就不再重复添加
    if item["id"].endswith(f"_{suffix}"):
        new_id = item["id"]
    else:
        new_id = f"{item['id']}_{suffix}"
    # 构造对话样本
    sft_data.append({
        "id": new_id,
        "image": item["img"],
        "conversations": [
            {
                "from": "human",
                "value": f'Does the image entail this statement:\n"{item["sentence"]}"'
            },
            {
                "from": "gpt",
                "value": "Yes" if item["label"] == 1 else "No"
            }
        ]
    })

# ===== 保存输出 =====
with open(output_file, "w") as f:
    json.dump(sft_data, f, indent=2)

print(f"✅ 转换完成，共 {len(sft_data)} 条样本，保存至 {output_file}")
