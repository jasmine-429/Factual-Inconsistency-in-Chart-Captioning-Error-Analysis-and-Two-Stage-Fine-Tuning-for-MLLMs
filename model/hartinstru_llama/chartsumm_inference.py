import os
import json
from tqdm import tqdm
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

# ===== 模型配置 =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "/data/jguo376/project/model/ChartInstruct-LLama2"

# ===== 路径配置 =====
input_jsons = [
    "/data/jguo376/project/dataset/chartsumm/test_k.json",
    "/data/jguo376/project/dataset/chartsumm/test_s.json"
]
output_jsons = [
    "/data/jguo376/project/model/hartinstru_llama/test_k_output.json",
    "/data/jguo376/project/model/hartinstru_llama/test_s_output.json"
]
image_root = "/data/jguo376/project/dataset/chartsumm/chart_images"

# ===== 加载模型 =====

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    trust_remote_code=True  # 🔥 关键点
).to(device)

processor = AutoProcessor.from_pretrained(model_id)

# ===== 推理函数 =====
def infer_chartinstruct(image_path, input_question="Please describe the chart."):
    image = Image.open(image_path).convert("RGB")
    prompt = f"<image>\n Question: {input_question} Answer: "

    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    inputs['pixel_values'] = inputs['pixel_values'].to(torch.float16)

    prompt_length = inputs['input_ids'].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            num_beams=4,
            max_new_tokens=512,
            early_stopping=True
        )

    output_text = processor.batch_decode(
        outputs[:, prompt_length:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )[0]
    return output_text.strip()

# ===== 主循环：批量处理 + 断点续跑 =====
for input_path, output_path in zip(input_jsons, output_jsons):
    print(f"\n📂 正在处理文件: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    # ==== 断点续跑：读取已完成样本 ====
    done_images = set()
    results = []
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            results = json.load(f)
            done_images = {item["image"] for item in results}
        print(f"🔄 已检测到 {len(done_images)} 条结果，将跳过已完成图像。")

    # ==== 批量处理 ====
    for item in tqdm(data_list, desc=f"Processing {os.path.basename(input_path)}"):
        image_name = item.get("image") or item.get("img")
        if image_name in done_images:
            continue

        image_path = os.path.join(image_root, image_name)
        if not os.path.exists(image_path):
            caption = f"[ERROR] Image not found: {image_path}"
        else:
            try:
                question = item.get("question") or item.get("input") or "Please describe the chart."
                caption = infer_chartinstruct(image_path, input_question=question)
            except Exception as e:
                caption = f"[ERROR] {str(e)}"

        result_item = {
            "image": image_name,
            "generated_caption": caption
        }
        results.append(result_item)

        # 每10条保存一次
        if len(results) % 10 == 0:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"[✓] {image_name}\n    → {caption}\n")

    # 最终保存完整结果
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✅ 推理完成，结果已保存至: {output_path}")
