#(/data/jguo376/conda_envs/qwen) (qwen) [jguo376@foscsmlprd01 script]$ python accuracy_lora.py 
#✅ Accuracy: 78.84% (518/657)
#(/data/jguo376/conda_envs/qwen) (qwen) [jguo376@foscsmlprd01 script]$ 
import json

# 读取文件
with open("/data/jguo376/project/dataset/test_dataset/ChartX/test_data/test_6600.json", "r", encoding="utf-8") as f:
    data = json.load(f)

correct = 0
total = 0

for item in data:
    # 标准答案（ground-truth）
    gt = None
    for conv in item["conversations"]:
        if conv["from"] == "gpt":
            gt = conv["value"].strip().lower()
            break
    if gt is None:
        continue  # 跳过无标注样本

    # 模型预测
    pred = item.get("model_prediction", "").strip().lower()
    pred_label = "yes" if pred.startswith("yes") else "no"
    true_label = "yes" if gt.startswith("yes") else "no"

    if pred_label == true_label:
        correct += 1
    total += 1

accuracy = correct / total if total > 0 else 0
print(f"✅ Accuracy: {accuracy:.2%} ({correct}/{total})")
