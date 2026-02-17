from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# 模型和图像路径
model_name = "ahmed-masry/unichart-base-960"
image_path = "/data/jguo376/project/dataset/ChartX_dataset/ChartX/bar_chart/png/bar_1.png"
input_prompt = "<summarize_chart> <s_answer>"

# 加载模型和处理器
model = VisionEncoderDecoderModel.from_pretrained(model_name)
processor = DonutProcessor.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 图像预处理
image = Image.open(image_path).convert("RGB")
decoder_input_ids = processor.tokenizer(input_prompt, add_special_tokens=False, return_tensors="pt").input_ids
pixel_values = processor(image, return_tensors="pt").pixel_values

# 推理生成
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

# 解码结果
sequence = processor.batch_decode(outputs.sequences)[0]
sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
caption = sequence.split("<s_answer>")[1].strip()

print("📝 Generated Chart Caption:", caption)
