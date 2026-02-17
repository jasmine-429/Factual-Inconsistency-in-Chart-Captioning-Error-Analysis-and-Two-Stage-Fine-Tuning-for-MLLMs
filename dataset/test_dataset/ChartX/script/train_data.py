import json

# ==== 文件路径配置 ====
annotation_path = "/data/jguo376/project/dataset/ChartX_dataset/ChartX/ChartX_annotation.json"
test_path = "/data/jguo376/project/dataset/test_dataset/ChartX/test_samples.json"
eval_path = "/data/jguo376/project/dataset/ChartX_dataset/ChartX/chart_evaluate.jsonl"
output_path = "//data/jguo376/project/dataset/test_dataset/ChartX/train_samples.json"

# ==== 读取 test_samples.json 中的 img 路径 ====
with open(test_path, "r") as f:
    test_data = json.load(f)
test_imgs = {item.get("img") for item in test_data}

# ==== 读取 chart_evaluate.jsonl 中的 img 路径 ====
eval_imgs = set()
with open(eval_path, "r") as f:
    for line in f:
        try:
            item = json.loads(line)
            eval_imgs.add(item.get("img"))
        except json.JSONDecodeError:
            continue

# ==== 读取总数据 ChartX_annotation.json ====
with open(annotation_path, "r") as f:
    all_data = json.load(f)

# ==== 筛选不在 test 和 eval 中的样本 ====
exclude_imgs = test_imgs.union(eval_imgs)
train_data = [item for item in all_data if item.get("img") not in exclude_imgs]

# ==== 保存训练集 ====
with open(output_path, "w") as f:
    json.dump(train_data, f, indent=2)

print(f"✅ 训练集筛选完成，共 {len(train_data)} 条，保存至: {output_path}")
