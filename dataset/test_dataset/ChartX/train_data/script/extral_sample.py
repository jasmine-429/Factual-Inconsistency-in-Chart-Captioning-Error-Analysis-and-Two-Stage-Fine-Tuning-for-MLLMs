import json
import random
from collections import defaultdict

# ===== 文件路径配置 =====
pos_file = "/data/jguo376/project/dataset/test_dataset/ChartX/test_data/dataset/test_samples_id.json"
value_error_file = "/data/jguo376/project/dataset/test_dataset/ChartX/test_data/dataset/error_data/value_error_augmented.json"
label_error_file = "/data/jguo376/project/dataset/test_dataset/ChartX/test_data/dataset/error_data/label_error_augmented.json"
trend_error_file = "/data/jguo376/project/dataset/test_dataset/ChartX/test_data/dataset/error_data/trend_errors.json"
ooc_error_file = "/data/jguo376/project/dataset/test_dataset/ChartX/test_data/dataset/error_data/ooc_error_augmented.json"
nonsense_error_file = "/data/jguo376/project/dataset/test_dataset/ChartX/test_data/dataset/error_data/nonsence_error_augmented.json"
output_file = "/data/jguo376/project/dataset/test_dataset/ChartX/test_data/dataset/test_balanced_mixed.json"

# ===== 错误类型比例配置 =====
error_ratios = {
    "value_error": 0.25,
    "label_error": 0.25,
    "trend_error": 0.15,
    "magnitude_error": 0.15,
    "ooc_error": 0.10,
    "nonsense_error": 0.10,
}

# ===== 加载正样本 =====
with open(pos_file) as f:
    pos_data = json.load(f)
random.shuffle(pos_data)
num_pos = 300  # ✅ 固定取 20000 个正样本
pos_data = pos_data[:num_pos]

# ===== 计算负样本目标数 =====
neg_target = int(num_pos * 1.2)
print(f"✅ 正样本数量: {num_pos}")
print(f"🎯 目标负样本数量: {neg_target}")

# ===== 加载负样本池 =====
with open(value_error_file) as f:
    value_data = json.load(f)
with open(label_error_file) as f:
    label_data = json.load(f)
with open(trend_error_file) as f:
    trend_data = json.load(f)
with open(ooc_error_file) as f:
    ooc_data = json.load(f)
with open(nonsense_error_file) as f:
    nonsense_data = json.load(f)

# trend_error 和 magnitude_error 分开
error_pool = defaultdict(list)
for item in value_data:
    error_pool["value_error"].append(item)
for item in label_data:
    error_pool["label_error"].append(item)
for item in trend_data:
    if item["error"] == "trend_error":
        error_pool["trend_error"].append(item)
    elif item["error"] == "magnitude_error":
        error_pool["magnitude_error"].append(item)
for item in ooc_data:
    error_pool["ooc_error"].append(item)
for item in nonsense_data:
    error_pool["nonsense_error"].append(item)

# ===== 分配负样本（不足就补）=====
neg_selected = []
actual_counts = {}
deficit = 0

# 阶段1：尝试按配额采样
for err_type, ratio in error_ratios.items():
    target_n = int(neg_target * ratio)
    pool = error_pool.get(err_type, [])
    random.shuffle(pool)
    if len(pool) >= target_n:
        selected = pool[:target_n]
    else:
        selected = pool
        deficit += (target_n - len(pool))
    actual_counts[err_type] = len(selected)
    neg_selected.extend(selected)

# 阶段2：补足不足部分（从有剩余的类型中填）
# 统计哪些还有富余
replenish_types = [k for k in error_ratios if len(error_pool[k]) > actual_counts.get(k, 0)]
remaining_total = sum([error_ratios[k] for k in replenish_types])
alloc = {k: int(deficit * (error_ratios[k] / remaining_total)) for k in replenish_types}

for err_type, extra_n in alloc.items():
    already_used = actual_counts[err_type]
    pool = error_pool[err_type][already_used:]
    random.shuffle(pool)
    supplement = pool[:extra_n]
    actual_counts[err_type] += len(supplement)
    neg_selected.extend(supplement)

# ===== 合并并保存 =====
final_data = pos_data + neg_selected
random.shuffle(final_data)

with open(output_file, "w") as f:
    json.dump(final_data, f, indent=2)

# ===== 打印统计 =====
print(f"\n✅ 最终样本总数: {len(final_data)}")
print(f"📊 各类负样本实际数量:")
for k, v in actual_counts.items():
    print(f"  {k:<16} : {v}")
