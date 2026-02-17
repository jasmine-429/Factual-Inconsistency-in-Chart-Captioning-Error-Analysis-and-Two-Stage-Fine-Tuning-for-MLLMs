import os
import json
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from PIL import Image
from tqdm import tqdm

torch.set_grad_enabled(False)
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# ===== 模型加载 =====
ckpt_path = "internlm/internlm-xcomposer2d5-7b"
config = AutoConfig.from_pretrained(ckpt_path, trust_remote_code=True)
config.attn_implementation = "eager"
tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
model = AutoModel.from_pretrained(
    ckpt_path,
    config=config,
    torch_dtype=torch.float16,
    trust_remote_code=True
).cuda().eval()
model.tokenizer = tokenizer

def ensure_rgb(image_path):
    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
        img.save(image_path)
    return image_path

def generate_caption(image_path, prompt):
    image = [ensure_rgb(image_path)]
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        response, _ = model.chat(
            tokenizer,
            prompt,
            image,
            do_sample=False,
            num_beams=1,
            max_new_tokens=256,
            use_meta=True
        )
    return response

# ===== 路径配置 =====
dataset_root = "/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/all_type/caption/deep_analysis/caption_all"
input_json = os.path.join(dataset_root, "internlm_missing_subset.json")
output_json = os.path.join(dataset_root, "internlm_caption_missing.json")
real_image_root = "/data/jguo376/project/dataset/ChartX_dataset/ChartX"
prompt = "Provide a short analytical description of the chart based on the data it shows."

# ===== 断点续跑：以 img 为唯一标识 =====
processed_imgs = set()
if os.path.exists(output_json):
    with open(output_json, "r", encoding="utf-8") as f:
        try:
            existing_data = json.load(f)
            for item in existing_data:
                processed_imgs.add(item["img"])
        except json.JSONDecodeError:
            existing_data = []
else:
    existing_data = []

results = existing_data.copy()

# ===== 参数配置 =====
batch_size = 6
save_every = 20
count = 0
max_test = None

def clear_cuda_cache():
    torch.cuda.empty_cache()

# ===== 加载数据 =====
with open(input_json, "r", encoding="utf-8") as f:
    data_list = json.load(f)

print("🚀 开始推理...")

for i in tqdm(range(0, len(data_list), batch_size)):
    if max_test is not None and count >= max_test:
        break

    batch_items = data_list[i:i + batch_size]
    for data in batch_items:
        if max_test is not None and count >= max_test:
            break

        rel_path = data["img"]
        image_path = os.path.join(real_image_root, rel_path.lstrip("./"))

        if image_path in processed_imgs:
            continue

        if not os.path.exists(image_path):
            caption = f"[ERROR] Image not found: {image_path}"
        else:
            try:
                caption = generate_caption(image_path, prompt)
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    caption = "[ERROR] CUDA out of memory"
                    clear_cuda_cache()
                else:
                    caption = f"[ERROR] {str(e)}"
            except Exception as e:
                caption = f"[ERROR] {str(e)}"

        output = {
            "chart_type": data.get("chart_type"),
            "imgname": data.get("imgname"),
            "img": image_path,
            "topic": data.get("topic"),
            "title": data.get("title"),
            "csv": data.get("csv"),
            "model_name": ckpt_path,
            "generated_caption": caption
        }
        results.append(output)
        processed_imgs.add(image_path)
        count += 1

        # 中间保存
        if count % save_every == 0:
            with open(output_json, "w", encoding="utf-8") as f_out:
                json.dump(results, f_out, ensure_ascii=False, indent=2)

    clear_cuda_cache()

# ===== 最终保存 =====
with open(output_json, "w", encoding="utf-8") as f_out:
    json.dump(results, f_out, ensure_ascii=False, indent=2)

print(f"\n✅ Done! Total: {len(results)} captions saved to: {output_json}")
