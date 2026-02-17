from PIL import Image
import requests
from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
torch.hub.download_url_to_file('https://raw.githubusercontent.com/vis-nlp/ChartQA/main/ChartQA%20Dataset/val/png/multi_col_1229.png', 'chart_example_1.png')

image_path = "/content/chart_example_1.png"
input_text = "What is the share of respondants who prefer Whatsapp in the 18-29 age group?"

input_prompt = f"<image>\n Question: {input_text} Answer: "

model_id = "ahmed-masry/ChartInstruct-LLama2"


model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    trust_remote_code=True  # 🔥 关键点
).to(device)

processor = AutoProcessor.from_pretrained(model_id)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

image = Image.open(image_path).convert('RGB')
inputs = processor(text=input_prompt, images=image, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

# change type if pixel_values in inputs to fp16. 
inputs['pixel_values'] = inputs['pixel_values'].to(torch.float16)
prompt_length = inputs['input_ids'].shape[1]

# move to device
inputs = {k: v.to(device) for k, v in inputs.items()}

# Generate
generate_ids = model.generate(**inputs, num_beams=4, max_new_tokens=512)
output_text = processor.batch_decode(generate_ids[:, prompt_length:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(output_text)
