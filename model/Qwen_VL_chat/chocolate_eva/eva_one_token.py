import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
import os

# ==== 配置路径 ====
model_path = "/data/jguo376/pretrained_models/Qwen2.5-VL-7B-Instruct"
image_path = "/data/jguo376/project/dataset/ChartX_dataset/ChartX/bar_chart/png/bar_123.png"
prompt_text = 'Does the image entail this statement: "The number of students in Science is higher than in Arts?"'

# ==== 加载模型和处理器 ====
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(model_path, device_map="cuda", torch_dtype=torch.float16, trust_remote_code=True).eval()

# ==== 构造 message 和 prompt ====
image = Image.open(image_path).convert("RGB")
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt_text}
    ]
}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# ==== 编码输入 ====
inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True).to("cuda")

# ==== 打印输入 token（含 assistant 开始位置） ====
print("🧾 Input Prompt:")
print(processor.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=False))

# ==== 前向传播获取 logits[-1] ====
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits  # shape: [1, seq_len, vocab_size]
    next_token_logits = logits[0, -1]  # 预测 assistant 第一个生成 token 的位置

# ==== 显示 top-10 token（基于 logits）====
print("\n🔮 Top-10 Predicted Tokens by logits[-1]:")
top_k = torch.topk(next_token_logits, k=10)
for i in range(10):
    tok_id = top_k.indices[i].item()
    tok = processor.tokenizer.decode([tok_id])
    print(f"{i+1}. Token: '{tok}' (id={tok_id}) | logit = {top_k.values[i].item():.4f}")

# ==== 真实生成 1 个 token ====
with torch.no_grad():
    generated = model.generate(**inputs, max_new_tokens=1)
    gen_token_id = generated[0, -1].item()
    gen_token = processor.tokenizer.decode([gen_token_id])

pred_token_id = torch.argmax(next_token_logits).item()
pred_token = processor.tokenizer.decode([pred_token_id])

print(f"\n🧪 generate(): '{gen_token}' | logits.argmax(): '{pred_token}'")
