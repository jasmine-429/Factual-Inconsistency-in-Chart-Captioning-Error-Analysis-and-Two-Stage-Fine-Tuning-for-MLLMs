import json
import random

with open("/data/jguo376/project/dataset/test_dataset/ChartX/sample__sft.json") as f:
    data = json.load(f)

random.shuffle(data)
split_idx = int(0.9 * len(data))
train_data = data[:split_idx]
val_data = data[split_idx:]

with open("/data/jguo376/project/dataset/test_dataset/train_test/dataset/train.json", "w") as f:
    json.dump(train_data, f, indent=2)

with open("/data/jguo376/project/dataset/test_dataset/train_test/dataset/val.json", "w") as f:
    json.dump(val_data, f, indent=2)

print(f"训练集: {len(train_data)} 条，验证集: {len(val_data)} 条")
