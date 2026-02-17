import os
import json
import shutil
from pathlib import Path
from tqdm import tqdm

# ===== 路径配置 =====
json_path = "/data/jguo376/project/dataset/ChartX_dataset/chartx.json"  # 🔁 你自己的 JSON 文件路径
image_base_dir = "/data/jguo376/project/dataset/ChartX_dataset/ChartX"  # 图像的根目录
output_dir = "/data/jguo376/project/mmca_images_all"  # 存储新图像的目录
os.makedirs(output_dir, exist_ok=True)

# ===== 命名规则 =====
def derive_new_name(rel_path: str) -> str:
    p = Path(rel_path)
    parts = p.parts
    chart_type = parts[0] if len(parts) >= 2 else ""
    base_name = Path(parts[-1]).stem
    ext = Path(parts[-1]).suffix
    return f"{chart_type}_{base_name}{ext}" if base_name.isdigit() else base_name + ext

# ===== 读取 JSON 并处理图像 =====
with open(json_path, "r") as f:
    data = json.load(f)

for item in tqdm(data):
    rel_img_path = item["img"].lstrip("./")  # 去掉 ./ 前缀
    abs_img_path = os.path.join(image_base_dir, rel_img_path)

    if not os.path.exists(abs_img_path):
        print(f"[⚠️ 跳过] 图像不存在: {abs_img_path}")
        continue

    new_name = derive_new_name(rel_img_path)
    target_path = os.path.join(output_dir, new_name)

    if not os.path.exists(target_path):
        shutil.copy2(abs_img_path, target_path)
        print(f"[✅ 拷贝] {abs_img_path} → {target_path}")
    else:
        print(f"[⏩ 已存在] 跳过 {target_path}")
