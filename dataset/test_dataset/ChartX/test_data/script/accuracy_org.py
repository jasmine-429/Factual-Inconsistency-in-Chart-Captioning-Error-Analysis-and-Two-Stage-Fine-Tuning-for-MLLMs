import json
from tqdm import tqdm
import re
from sklearn.metrics import accuracy_score, classification_report

# ===== 配置路径 =====
input_json = "/data/jguo376/project/dataset/test_dataset/ChartX/test_data/test_entail_org.json"

# ===== 加载数据 =====
with open(input_json, 'r') as f:
    data = json.load(f)

gold_labels = []
pred_labels = []

def normalize_answer(ans: str) -> str:
    """从模型输出中解析 Yes / No 标签"""
    ans = ans.lower()
    if re.search(r"\b(no|not|don't|doesn't|cannot|can not|fail|unable)\b", ans):
        return "no"
    if ans.strip().startswith("no"):
        return "no"
    if ans.strip().startswith("yes"):
        return "yes"
    return "yes"  # fallback：默认是 entailment

for item in tqdm(data):
    # 提取标准答案（来自 conversation 中 GPT 回复）
    conv = item.get("conversations", [])
    gold = None
    for turn in conv:
        if turn["from"] == "gpt":
            gold = turn["value"].strip().lower()
            break
    if gold not in {"yes", "no"}:
        continue  # 跳过无效条目

    # 提取模型输出
    model_pred = item.get("model_prediction", "")
    pred = normalize_answer(model_pred)

    gold_labels.append(gold)
    pred_labels.append(pred)

# ===== 计算准确率与报告 =====
acc = accuracy_score(gold_labels, pred_labels)
report = classification_report(gold_labels, pred_labels, digits=3)

print(f"\n✅ Accuracy: {acc:.4f}")
print("\n🔍 Classification Report:")
print(report)
