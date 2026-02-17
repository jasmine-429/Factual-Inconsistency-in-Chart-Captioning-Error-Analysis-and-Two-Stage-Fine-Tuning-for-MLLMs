import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 设置模型名称和图像路径
model_id = "Qwen/Qwen-VL-Chat"
image_path = "/data/jguo376/project/dataset/ChartX_dataset/ChartX/bar_chart/png/bar_34.png"

# 加载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",  # 或 "cpu"
    trust_remote_code=True,
    bf16=True,
    ignore_mismatched_sizes=True
).eval()

# 构造推理输入
query = tokenizer.from_list_format([
    {"image": image_path},
    {"text": "Provide a short analytical description of the chart based on the data it shows."}
])

# 执行推理
response, _ = model.chat(tokenizer, query=query, history=None)

# 输出结果
print("📝 图表描述结果：")
print(response)
