import json
import re
from tqdm import tqdm

# ==== 输入输出路径 ====
input_path = "/data/jguo376/project/dataset/test_dataset/ChartX/test_data/dataset/test_sentences.json"
output_path = "/data/jguo376/project/dataset/test_dataset/ChartX/test_data/dataset/test_add_imgname.json"

def needs_fixing(imgname):
    """判断是否为纯数字 imgname"""
    return bool(re.fullmatch(r"\d+", imgname))

def fix_imgname(chart_type, imgname):
    """根据 chart_type 修复 imgname，如 85 → bar_85"""
    prefix = chart_type.split("_")[0]
    return f"{prefix}_{imgname}"

# ==== 加载数据 ====
with open(input_path, "r") as f:
    data = json.load(f)

# ==== 修正 imgname 字段 ====
for item in tqdm(data, desc="Fixing imgname"):
    imgname = item.get("imgname", "")
    chart_type = item.get("chart_type", "")
    
    if needs_fixing(imgname):
        item["imgname"] = fix_imgname(chart_type, imgname)

# ==== 保存结果 ====
with open(output_path, "w") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"✅ 已处理 {len(data)} 条样本，imgname 修复完成，保存到：{output_path}")
