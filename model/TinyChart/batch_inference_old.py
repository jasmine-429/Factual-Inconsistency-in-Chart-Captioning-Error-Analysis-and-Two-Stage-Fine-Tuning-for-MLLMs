import os
import json
from tqdm import tqdm
from PIL import Image
from tinychart.model.builder import load_pretrained_model
from tinychart.mm_utils import get_model_name_from_path
from tinychart.eval.run_tiny_chart import inference_model

# ========= 环境配置 =========
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
model_path = "mPLUG/TinyChart-3B-768"
device = "cuda:0"

# ========= 模型加载 =========
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path, 
    model_base=None,
    model_name=get_model_name_from_path(model_path),
    device=device
)

# ========= 输入输出路径配置 =========
dataset_root = "/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/all_type/caption/deep_analysis/caption_all"
input_jsonl = os.path.join(dataset_root, "chartx_selected_fields.json")
output_json = os.path.join(dataset_root, "tinychart_caption_output.json")
chart_root = "/data/jguo376/project/dataset/ChartX_dataset/ChartX"
prompt = "Provide a short analytical description of the chart, including specific values, comparisons, and trends."

# ========= 加载输入数据 =========
with open(input_jsonl, "r", encoding="utf-8") as f:
    data_list = [json.loads(line) for line in f.readlines()]

results = []
max_test = None  # 如只想测试前 3 个样本，可设为 3；否则设为 None 表示全部处理

# ========= 推理并生成结果 =========
print("🚀 Start TinyChart caption generation...")
count = 0
for item in tqdm(data_list, desc="Generating captions"):
    if max_test is not None and count >= max_test:
        break

    imgname = item["imgname"]
    rel_path = item["img"]
    image_path = os.path.join(chart_root, rel_path.replace("./", ""))

    if not os.path.exists(image_path):
        caption = f"[ERROR] Image not found: {image_path}"
    else:
        try:
            caption = inference_model([image_path], prompt, model, tokenizer, image_processor, context_len, conv_mode="phi", max_new_tokens=256)
        except Exception as e:
            caption = f"[ERROR] {str(e)}"

    item["model_name"] = model_path
    item["img"] = image_path
    item["generated_caption"] = caption
    results.append(item)
    count += 1

    print(f"[✓] {imgname}")
    print(f"    → {caption}\n")

# ========= 保存输出结果 =========
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n✅ TinyChart 推理完成！共处理 {len(results)} 张图，输出保存至: {output_json}")
