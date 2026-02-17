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

# ===== 路径配置 =====
input_jsons = [
    "/data/jguo376/project/dataset/chartsumm/test_k.json",
    "/data/jguo376/project/dataset/chartsumm/test_s.json"
]
output_jsons = [
    "/data/jguo376/project/model/Internlm-xcomposer2-vl-7b/test_k_output.json",
    "/data/jguo376/project/model/Internlm-xcomposer2-vl-7b/test_s_output.json"
]
image_root = "/data/jguo376/project/dataset/chartsumm/chart_images"
prompt = "Please describe the chart."

# ===== 功能函数 =====
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

def clear_cuda_cache():
    torch.cuda.empty_cache()

# ===== 主处理流程 =====
for input_json, output_json in zip(input_jsons, output_jsons):
    print(f"\n🚀 开始处理：{input_json}")

    # 加载已有结果（断点续跑）
    processed_imgs = set()
    if os.path.exists(output_json):
        with open(output_json, "r", encoding="utf-8") as f:
            try:
                existing_data = json.load(f)
                for item in existing_data:
                    processed_imgs.add(item["image"])
            except json.JSONDecodeError:
                existing_data = []
    else:
        existing_data = []

    results = existing_data.copy()
    count = 0
    save_every = 20
    batch_size = 6
    max_test = None

    with open(input_json, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    for i in tqdm(range(0, len(data_list), batch_size)):
        if max_test is not None and count >= max_test:
            break

        batch_items = data_list[i:i + batch_size]
        for data in batch_items:
            if max_test is not None and count >= max_test:
                break

            rel_path = data["image"]
            image_path = os.path.join(image_root, rel_path.lstrip("./"))

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
                "image": image_path,
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

    # 最终保存
    with open(output_json, "w", encoding="utf-8") as f_out:
        json.dump(results, f_out, ensure_ascii=False, indent=2)

    print(f"✅ 生成完成：{len(results)} captions saved to: {output_json}")
