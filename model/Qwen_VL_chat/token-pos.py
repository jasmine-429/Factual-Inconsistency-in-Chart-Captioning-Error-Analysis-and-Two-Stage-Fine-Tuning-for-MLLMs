from transformers import AutoTokenizer

#'yes' token ID(s): [9693]
#'no' token ID(s): [2152]

# 替换为你自己的模型路径或 HuggingFace 名字
model_name_or_path = "/data/jguo376/pretrained_models/Qwen2.5-VL-7B-Instruct"

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

# 编码 "yes" 和 "no"
yes_token_ids = tokenizer("yes", add_special_tokens=False).input_ids
no_token_ids = tokenizer("no", add_special_tokens=False).input_ids

# 打印 Token ID
print(f"'yes' token ID(s): {yes_token_ids}")
print(f"'no' token ID(s): {no_token_ids}")
