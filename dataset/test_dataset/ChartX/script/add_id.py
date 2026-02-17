import json
from tqdm import tqdm
from collections import defaultdict, OrderedDict

# ==== 输入输出路径 ====
input_path = "/data/jguo376/project/dataset/test_dataset/ChartX/test_data/dataset/test_add_imgname.json"
output_path = "/data/jguo376/project/dataset/test_dataset/ChartX/test_data/dataset/test_samples_id.json"

# ==== 加载数据 ====
with open(input_path, "r") as f:
    data = json.load(f)

# ==== 统计 imgname 出现次数，并插入 ID ====
imgname_counter = defaultdict(int)
results = []

for item in tqdm(data, desc="Assigning ordered IDs"):
    imgname = item.get("imgname", "")
    imgname_counter[imgname] += 1
    sentence_index = imgname_counter[imgname]
    sent_id = f"{imgname}_{sentence_index}"

    # 保证顺序：chart_type, img, imgname, id, 其他字段...
    ordered_item = OrderedDict()
    ordered_item["chart_type"] = item.get("chart_type", "")
    ordered_item["img"] = item.get("img", "")
    ordered_item["imgname"] = imgname
    ordered_item["id"] = sent_id

    for key in item:
        if key not in ordered_item:
            ordered_item[key] = item[key]

    results.append(ordered_item)

# ==== 保存为 JSON ====
with open(output_path, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"✅ ID 已按顺序插入，共处理 {len(results)} 条，保存至：{output_path}")
