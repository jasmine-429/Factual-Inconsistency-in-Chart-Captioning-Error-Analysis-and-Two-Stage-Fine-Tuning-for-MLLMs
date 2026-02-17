#分割训练数据集
import json
import os

# ====== 配置 ======
input_file = "/data/jguo376/project/dataset/test_dataset/ChartX/train_data/data_42000/chart_entail_sharegpt.json"
output_dir = "/data/jguo376/project/llama_factory/data"  # ✅ 输出文件夹
output_prefix = "chart_entail_part"
num_parts = 7

# ====== 创建输出文件夹（如不存在） ======
os.makedirs(output_dir, exist_ok=True)

# ====== 加载数据 ======
with open(input_file, "r") as f:
    data = json.load(f)

total = len(data)
part_size = total // num_parts

# ====== 分割写入 ======
for i in range(num_parts):
    start = i * part_size
    end = (i + 1) * part_size if i < num_parts - 1 else total
    part_data = data[start:end]

    output_file = os.path.join(output_dir, f"{output_prefix}_{i}.json")  # ✅ 拼接完整路径
    with open(output_file, "w") as f:
        json.dump(part_data, f, indent=2)

    print(f"✅ Wrote {len(part_data)} samples to {output_file}")
