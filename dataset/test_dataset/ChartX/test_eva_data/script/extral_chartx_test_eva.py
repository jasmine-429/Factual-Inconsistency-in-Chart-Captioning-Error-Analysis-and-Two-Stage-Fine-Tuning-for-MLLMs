#提取验证和测试集
import json

# === 路径配置 ===
anno_path = "/Users/gjx/Desktop/Paper/dataset/ChaartX/ChartX/ChartX_annotation.json"
train_path = "/Users/gjx/Desktop/test_dataset/ChartX/train_samples_with_QA.json"
output_path = "/Users/gjx/Desktop/test_dataset/ChartX/train_eva_data/data/eva_test.json"

# === 加载数据 ===
with open(anno_path, "r") as f:
    annotation_data = json.load(f)

with open(train_path, "r") as f:
    train_data = json.load(f)

# === 提取 train 中的 img 集合（唯一项）===
train_imgs = set(item["img"] for item in train_data)

# === 提取 annotation 中未出现在 train 中的样本 ===
missing_data = []
for item in annotation_data:
    if item["img"] not in train_imgs:
        # 只保留指定字段
        filtered = {k: item[k] for k in ["chart_type", "img", "imgname", "csv", "title", "topic"] if k in item}
        missing_data.append(filtered)

# === 保存结果 ===
with open(output_path, "w") as f:
    json.dump(missing_data, f, indent=2)

print(f"✅ 共提取 {len(missing_data)} 条样本（以 img 为唯一键），已保存至：{output_path}")
