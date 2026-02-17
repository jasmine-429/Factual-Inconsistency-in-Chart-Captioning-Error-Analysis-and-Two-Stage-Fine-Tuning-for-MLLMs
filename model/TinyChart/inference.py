
import torch
from PIL import Image
from tinychart.model.builder import load_pretrained_model
from tinychart.mm_utils import get_model_name_from_path
from tinychart.eval.run_tiny_chart import inference_model
from tinychart.eval.eval_metric import parse_model_output, evaluate_cmds
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

def show_image(img_path):
    img = Image.open(img_path).convert('RGB')
    #img.show()

# Build the model
model_path = "/data/jguo376/pretrained_models/TinyChart-3B-768"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path, 
    model_base=None,
    model_name=get_model_name_from_path(model_path),
    device="cuda:0" # device="cpu" if running on cpu
)


img_path = "/data/jguo376/project/dataset/ChartX_dataset/ChartX/bar_chart/png/bar_5.png"
show_image(img_path)

# To use program-of-thoughts, append `Answer with detailed steps.` to the prompt
text = "Write a concise paragraph that describes the chart, including key values, categories, and noticeable trends."

response = inference_model([img_path], text, model, tokenizer, image_processor, context_len, conv_mode="phi", max_new_tokens=1024)

# Run this code to evaluate the generated python code
print("📝 Model Output:")
print(response)