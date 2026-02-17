#小样本微调的采样脚本一共采样220条
import json
import random
from collections import defaultdict

# ===== 路径配置 =====
pos_file = "/data/jguo376/project/dataset/test_dataset/ChartX/train/train_samples_id.json"
value_error_file = "/data/jguo376/project/dataset/test_dataset/ChartX/train/value_error_augmented.json"
label_error_file = "/data/jguo376/project/dataset/test_dataset/ChartX/train/label_error_augmented.json"
trend_error_file = "/data/jguo376/project/dataset/test_dataset/ChartX/train/train_trend_errors.json"
ooc_error_file = "/data/jguo376/project/dataset/test_dataset/ChartX/train/ooc_error_augmented.json"
nonsense_error_file = "/data/jguo376/project/dataset/test_dataset/ChartX/train/nonsence_error_augmented.json"
output_file = "/data/jguo376/project/dataset/test_dataset/ChartX/trainsample_200_mixed.json"

# ===== 优先图表类型与采样数量配置 =====
priority_charts = {"bar_chart", "line_chart", "pie_chart", "bar_chart_num", "line_chart_num"}
per_error_limit = 2  # 每种错误类型采样上限

# ===== 有 method 字段的采样函数（用于 ooc 和 nonsense）=====
def sample_with_priority_and_method(data, error_type, method_field=False):
    method_buckets = defaultdict(list)
    for item in data:
        if item["error"] != error_type:
            continue
        method = item.get("method", "unknown") if method_field else "default"
        method_buckets[method].append(item)

    selected = []
    for items in method_buckets.values():
        priority = [x for x in items if x["chart_type"] in priority_charts]
        others = [x for x in items if x["chart_type"] not in priority_charts]
        random.shuffle(priority)
        random.shuffle(others)
        k = max(1, per_error_limit // len(method_buckets))
        selected += priority[:k] + others[: (per_error_limit - len(selected))]
        if len(selected) >= per_error_limit:
            break
    return selected[:per_error_limit]

# ===== 普通错误采样函数（value、label、trend、magnitude）=====
def sample_simple(data, error_type):
    priority = [x for x in data if x["chart_type"] in priority_charts and x["error"] == error_type]
    others = [x for x in data if x["chart_type"] not in priority_charts and x["error"] == error_type]
    random.shuffle(priority)
    random.shuffle(others)
    return (priority[:int(per_error_limit * 0.5)] + others[:int(per_error_limit * 0.5)])[:per_error_limit]

# ===== 加载数据 =====
with open(pos_file) as f:
    pos_data = json.load(f)
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

# ===== 采样正样本 =====
pos_priority = [x for x in pos_data if x["chart_type"] in priority_charts]
pos_others = [x for x in pos_data if x["chart_type"] not in priority_charts]
random.shuffle(pos_priority)
random.shuffle(pos_others)
pos_selected = pos_priority[:5] + pos_others[:5]  # 共 100 条正样本

# ===== 采样负样本（确保每种错误类型都覆盖）=====
neg_selected = []
neg_selected += sample_simple(value_data, "value_error")
neg_selected += sample_simple(label_data, "label_error")
neg_selected += sample_simple(trend_data, "trend_error")
neg_selected += sample_simple(trend_data, "magnitude_error")
neg_selected += sample_with_priority_and_method(ooc_data, "ooc_error", method_field=True)
neg_selected += sample_with_priority_and_method(nonsense_data, "nonsense_error", method_field=True)

# ===== 合并与保存 =====
final_data = pos_selected + neg_selected
random.shuffle(final_data)

with open(output_file, "w") as f:
    json.dump(final_data, f, indent=2)

print(f"✅ 成功保存 {len(final_data)} 条样本到 {output_file}")
