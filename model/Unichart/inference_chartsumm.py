import os
import json
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
from tqdm import tqdm

# ===== 模型与环境配置 =====
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_name = "ahmed-masry/unichart-chart2text-statista-960"
processor = DonutProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

input_prompt = "<summarize_chart> <s_answer>"

# ===== 路径配置 =====
image_root = "/data/jguo376/project/dataset/chartsumm/chart_images"
input_jsons = [
    "/data/jguo376/project/dataset/chartsumm/test_k.json",
    "/data/jguo376/project/dataset/chartsumm/test_s.json"
]
output_jsons = [
    "/data/jguo376/project/model/Unichart/test_k_output.json",
    "/data/jguo376/project/model/Unichart/test_s_output.json"
]

# ===== 推理函数 =====
def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    decoder_input_ids = processor.tokenizer(input_prompt, add_special_tokens=False, return_tensors="pt").input_ids
    pixel_values = processor(image, return_tensors="pt").pixel_values

    outputs = model.generate(
        pixel_values.to(device),
        decoder_input_ids=decoder_input_ids.to(device),
        max_length=model.config.decoder.max_position_embeddings,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=4,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )
    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    return sequence.split("<s_answer>")[-1].strip()

# ===== 主循环：处理多个文件 =====
for input_json, output_json in zip(input_jsons, output_jsons):
    print(f"\n🚀 Start caption generation for: {input_json}")

    # 加载输入数据
    with open(input_json, "r", encoding="utf-8") as infile:
        data_list = json.load(infile)

    # 加载已完成结果
    processed_imgs = set()
    if os.path.exists(output_json):
        with open(output_json, "r", encoding="utf-8") as f:
            try:
                existing_results = json.load(f)
                for entry in existing_results:
                    processed_imgs.add(entry["image"])
            except json.JSONDecodeError:
                existing_results = []
    else:
        existing_results = []

    results = existing_results.copy()
    max_test = None
    save_every = 20
    count = 0

    for data in tqdm(data_list, desc=f"Generating captions for {os.path.basename(input_json)}"):
        image_name = data["image"]
        image_path = os.path.join(image_root, image_name)

        if image_name in processed_imgs:
            continue
        if max_test is not None and count >= max_test:
            break

        if not os.path.exists(image_path):
            caption = f"[ERROR] Image not found: {image_path}"
        else:
            try:
                caption = generate_caption(image_path)
            except Exception as e:
                caption = f"[ERROR] {str(e)}"

        output = {
            "image": image_name,
            "generated_caption": caption
        }

        results.append(output)
        processed_imgs.add(image_name)
        count += 1

        print(f"[✓] {image_name}")
        print(f"    → {caption}\n")

        if count % save_every == 0:
            with open(output_json, "w", encoding="utf-8") as f_out:
                json.dump(results, f_out, ensure_ascii=False, indent=2)

        torch.cuda.empty_cache()

    # 保存最终结果
    with open(output_json, "w", encoding="utf-8") as f_out:
        json.dump(results, f_out, ensure_ascii=False, indent=2)

    print(f"\n✅ Done! Total: {len(results)} captions saved to: {output_json}")
