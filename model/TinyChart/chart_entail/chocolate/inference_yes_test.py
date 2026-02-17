import sys
sys.path.append("/data/jguo376/project/model/TinyChart")

import torch
import os
import json
from tqdm import tqdm
from PIL import Image
from tinychart.model.builder import load_pretrained_model
from tinychart.mm_utils import get_model_name_from_path
from tinychart.eval.run_tiny_chart import inference_model

# ========= 配置路径 =========
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

model_path = "/data/jguo376/pretrained_models/TinyChart-3B-768"
data_path = "/data/jguo376/project/dataset/chcolate/CHOCOLATE/chocolate_test.jsonl"
save_path = "/data/jguo376/project/model/TinyChart/chart_entail/chocolate/800/org_yes_no/inference_model_raw_outputs.jsonl"

# ========= 加载模型 =========
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path),
    device="cuda:0"
)

# ========= 构造 prompt（符合 inference_model 接口格式） =========
def format_prompt(sentence):
    return f'Does the image entail the following statement? "{sentence}" Answer with Yes or No.'

# ========= 批量推理 =========
with open(data_path, "r") as f:
    samples = [json.loads(line) for line in f]

with open(save_path, "w") as fout:
    for sample in tqdm(samples):
        image_path = sample["local_image_path"]
        try:
            # 打开图像
            Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"[Image Error] {image_path}: {e}")
            continue

        for i, sentence in enumerate(sample["sentences"]):
            prompt = format_prompt(sentence)
            try:
                response = inference_model(
                    [image_path],
                    prompt,
                    model, tokenizer, image_processor, context_len,
                    conv_mode="phi",
                    max_new_tokens=10
                )
                gen_text = response[0]
            except Exception as e:
                print(f"[Generation Error] {sample['_id']} sent {i}: {e}")
                gen_text = "[Generation Failed]"

            result = {
                "_id": sample["_id"],
                "sentence_id": f"{sample['_id']}_{i}",
                "sentence": sentence,
                "prompt": prompt,
                "model_output": gen_text  # 不做解析
            }
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
