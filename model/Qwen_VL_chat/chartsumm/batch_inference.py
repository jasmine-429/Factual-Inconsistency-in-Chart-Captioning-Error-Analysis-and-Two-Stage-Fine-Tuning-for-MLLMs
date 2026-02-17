import os
import json
import traceback
from PIL import Image
from tqdm import tqdm
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from peft import PeftModel

# ========= 配置 =========
torch.set_grad_enabled(False)

model_path = "/data/jguo376/pretrained_models/Qwen2.5-VL-7B-Instruct"
lora_path = "/data/jguo376/project/llama_factory/saves/Qwen2-VL-7B-Instruct/lora/chartsumm_sample/checkpoint-700"  # ✅ LoRA 权重路径
use_lora = True  # ✅ 设置 True 可加载 LoRA

input_jsons = [
    "/data/jguo376/project/dataset/chartsumm/test_k.json",
    "/data/jguo376/project/dataset/chartsumm/test_s.json"
]
output_jsons = [
    "/data/jguo376/project/model/Qwen_VL_chat/chartsumm/chartsumm_single_800/test_k_output.json",
    "/data/jguo376/project/model/Qwen_VL_chat/chartsumm/chartsumm_single_800/test_s_output.json"
]
base_dir = "/data/jguo376/project/dataset/chartsumm/chart_images/"

max_items = None  # ✅ 设置为整数（如 10）调试用；None 表示处理全部

# ========= 加载模型与 Processor =========
base_model = AutoModelForVision2Seq.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
).eval()

model = PeftModel.from_pretrained(base_model, lora_path).eval() if use_lora else base_model
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

# ========= 推理函数 =========
def run_inference(input_json, output_json, base_dir, max_items=None):
    if os.path.exists(output_json):
        with open(output_json, "r", encoding="utf-8") as f:
            try:
                existing_results = json.load(f)
            except json.JSONDecodeError:
                existing_results = []
        existing_imgs = set(item["image"] for item in existing_results)
    else:
        existing_results = []
        existing_imgs = set()

    results = existing_results.copy()

    with open(input_json, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    for idx, item in enumerate(tqdm(data_list, desc=f"Generating captions for {os.path.basename(input_json)}")):
        if max_items is not None and idx >= max_items:
            print(f"🚫 达到最大处理数 {max_items}，提前停止")
            break

        image_name = item["image"]
        img_path = os.path.join(base_dir, image_name)

        if image_name in existing_imgs:
            continue
        if not os.path.exists(img_path):
            print(f"⚠️ 图像不存在: {img_path}")
            continue

        try:
            image = Image.open(img_path).convert("RGB")
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Please describe the chart."}]}]
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            inputs = processor(text=[prompt], images=[image], return_tensors="pt").to(model.device)

            output_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                eos_token_id=processor.tokenizer.eos_token_id,
                pad_token_id=processor.tokenizer.pad_token_id
            )

            caption_raw = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
            caption = caption_raw.split("assistant\n", 1)[-1].strip()

            results.append({
                "image": image_name,
                "generated_caption": caption
            })

            # ✅ 实时保存（断点续跑）
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"❌ 错误处理样本 {idx}（{image_name}）: {e}")
            traceback.print_exc()

    print(f"✅ 推理完成，结果保存在：{output_json}")

# ========= 执行两个文件的推理 =========
for inp, out in zip(input_jsons, output_jsons):
    run_inference(inp, out, base_dir, max_items=max_items)

