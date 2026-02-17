import os
import sys
import json
import torch
from PIL import Image
from tqdm import tqdm
import traceback

# ========= 环境设置 =========
sys.path.append("/data/jguo376/project/model/mPLUG-Owl/mPLUG-Owl")
torch.set_grad_enabled(False)

from transformers import AutoTokenizer
from mplug_owl.modeling_mplug_owl import MplugOwlForConditionalGeneration
from mplug_owl.processing_mplug_owl import MplugOwlProcessor, MplugOwlImageProcessor
from peft import PeftModel

# ========= 路径配置 =========
model_path_for_weights = "/data/jguo376/pretrained_models/mmca_merged_model"
processor_ref_path = "/data/jguo376/pretrained_models/mplug-owl-llama-7b"
lora_path = "/data/jguo376/project/model/MMCA/fine-tuning/output/sft_v0.1_ft_chartsumm_caption_prompt/checkpoint-1500"

input_jsons = [
    "/data/jguo376/project/dataset/chartsumm/test_k.json",
    "/data/jguo376/project/dataset/chartsumm/test_s.json"
]
output_jsons = [
    "/data/jguo376/project/model/MMCA/chartsumm_differnet_fine/1500_new/test_k_output.json",
    "/data/jguo376/project/model/MMCA/chartsumm_differnet_fine/1500_new/test_s_output.json"
]
chart_root = "/data/jguo376/project/dataset/chartsumm/chart_images/"
max_items = None  # None 表示全部处理

# ========= 加载模型 =========
print("📦 加载基础模型权重...")
model = MplugOwlForConditionalGeneration.from_pretrained(
    model_path_for_weights,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
if lora_path:
    print(f"🪄 加载 LoRA adapter: {lora_path}")
    model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    # 🔥 关键验证步骤
    print("🔍 验证LoRA是否生效...")
    
    # 检查LoRA模块
    lora_modules = [name for name, module in model.named_modules() 
                   if hasattr(module, 'lora_A')]
    
    if lora_modules:
        print(f"✅ 找到 {len(lora_modules)} 个LoRA模块")
        print(f"   示例模块: {lora_modules[0]}")
        
        # 🔥 修复后的权重数值检查
        print("🔍 检查LoRA权重数值...")
        total_lora_norm = 0
        zero_count = 0
        checked_count = 0
        
        for name, module in model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                try:
                    # 处理不同的LoRA结构
                    if hasattr(module.lora_A, 'weight'):
                        lora_A_norm = module.lora_A.weight.norm().item()
                    elif hasattr(module.lora_A, 'default'):
                        lora_A_norm = module.lora_A.default.weight.norm().item()
                    else:
                        # 如果是ModuleDict，尝试获取default key
                        if isinstance(module.lora_A, torch.nn.ModuleDict):
                            if 'default' in module.lora_A:
                                lora_A_norm = module.lora_A['default'].weight.norm().item()
                            else:
                                # 获取第一个key
                                first_key = list(module.lora_A.keys())[0]
                                lora_A_norm = module.lora_A[first_key].weight.norm().item()
                        else:
                            continue
                    
                    if hasattr(module.lora_B, 'weight'):
                        lora_B_norm = module.lora_B.weight.norm().item()
                    elif hasattr(module.lora_B, 'default'):
                        lora_B_norm = module.lora_B.default.weight.norm().item()
                    else:
                        if isinstance(module.lora_B, torch.nn.ModuleDict):
                            if 'default' in module.lora_B:
                                lora_B_norm = module.lora_B['default'].weight.norm().item()
                            else:
                                first_key = list(module.lora_B.keys())[0]
                                lora_B_norm = module.lora_B[first_key].weight.norm().item()
                        else:
                            continue
                    
                    total_lora_norm += lora_A_norm + lora_B_norm
                    
                    if lora_A_norm < 1e-6 and lora_B_norm < 1e-6:
                        zero_count += 1
                    
                    # 只打印前5个模块的详细信息
                    if checked_count < 5:
                        print(f"  {name}:")
                        print(f"    lora_A norm: {lora_A_norm:.8f}")
                        print(f"    lora_B norm: {lora_B_norm:.8f}")
                        # 打印LoRA结构信息
                        print(f"    lora_A type: {type(module.lora_A)}")
                        print(f"    lora_B type: {type(module.lora_B)}")
                    
                    checked_count += 1
                    
                except Exception as e:
                    if checked_count < 3:  # 只打印前几个错误
                        print(f"  ❌ 检查 {name} 时出错: {e}")
                        print(f"    lora_A type: {type(module.lora_A)}")
                        print(f"    lora_B type: {type(module.lora_B)}")
                        # 尝试打印结构
                        if isinstance(module.lora_A, torch.nn.ModuleDict):
                            print(f"    lora_A keys: {list(module.lora_A.keys())}")
                        if isinstance(module.lora_B, torch.nn.ModuleDict):
                            print(f"    lora_B keys: {list(module.lora_B.keys())}")
                    continue
        
        print(f"\n📊 LoRA权重统计:")
        print(f"  成功检查的模块: {checked_count}/{len(lora_modules)}")
        if checked_count > 0:
            print(f"  总权重范数: {total_lora_norm:.8f}")
            print(f"  接近零的模块: {zero_count}/{checked_count}")
            print(f"  平均权重范数: {total_lora_norm/checked_count:.8f}")
            
            if total_lora_norm < 1e-3:
                print("❌ 严重问题：所有LoRA权重都接近零！训练没有生效！")
            elif zero_count > checked_count * 0.8:
                print("⚠️ 警告：大部分LoRA权重接近零，训练可能不充分")
            else:
                print("✅ LoRA权重看起来正常")
        else:
            print("❌ 无法检查任何LoRA权重")
            
        # 🔥 检查checkpoint文件大小
        adapter_file = os.path.join(lora_path, "adapter_model.bin")
        if os.path.exists(adapter_file):
            file_size = os.path.getsize(adapter_file) / 1024 / 1024  # MB
            print(f"📁 adapter_model.bin 大小: {file_size:.2f} MB")
            if file_size < 1.0:
                print("⚠️ 警告：adapter文件很小，可能权重更新不充分")
            else:
                print("✅ adapter文件大小正常")
        else:
            print(f"❌ 未找到 adapter_model.bin 文件")
    else:
        print("❌ 警告：未找到LoRA模块！")
# ========= 加载 processor =========
tokenizer = AutoTokenizer.from_pretrained(processor_ref_path)
image_processor = MplugOwlImageProcessor.from_pretrained(processor_ref_path)
processor = MplugOwlProcessor(image_processor, tokenizer)

# ========= 推理参数 =========
generate_kwargs = {
    'do_sample': True,
    'top_k': 5,
    'max_length': 512
}

# ========= 推理函数 =========
def run_inference(input_json, output_json, chart_root, query_prompt, max_items=None):
    print(f"\n📂 处理文件：{input_json}")
    
    if os.path.exists(output_json):
        with open(output_json, "r", encoding="utf-8") as f:
            try:
                existing_results = json.load(f)
            except json.JSONDecodeError:
                existing_results = []
        processed_imgs = set(item["image"] for item in existing_results)
    else:
        existing_results = []
        processed_imgs = set()

    results = existing_results.copy()

    with open(input_json, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    if max_items is not None:
        data_list = data_list[:max_items]

    for idx, item in enumerate(tqdm(data_list, desc=f"Generating captions for {os.path.basename(input_json)}")):
        image_name = item.get("image") or item.get("img")
        img_path = os.path.join(chart_root, image_name)

        if image_name in processed_imgs:
            continue
        if not os.path.exists(img_path):
            print(f"⚠️ 图像不存在: {img_path}")
            continue

        try:
            image = Image.open(img_path).convert("RGB")
            inputs = processor(text=[query_prompt], images=[image], return_tensors="pt")
            inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                output = model.generate(**inputs, **generate_kwargs)
                caption = tokenizer.decode(output[0], skip_special_tokens=True)
        except Exception as e:
            print(f"❌ 错误：{image_name} - {e}")
            traceback.print_exc()
            caption = f"[ERROR] {str(e)}"

        result = {
            "image": image_name,
            "generated_caption": caption
        }
        results.append(result)
        processed_imgs.add(image_name)

        print(f"[✓] {image_name}")
        print(f"    → {caption}\n")

        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ 完成：{output_json}，共生成 {len(results)} 条描述")


# ========= 分别设置 prompt 并处理 =========
query_prompts = [
    """The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
Human: <image>
Human: Please generate a short summary of the chart.
AI:""",
    """The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
Human: <image>
Human: Please generate a long summary of the chart.
AI:"""
]

for in_path, out_path, prompt in zip(input_jsons, output_jsons, query_prompts):
    run_inference(in_path, out_path, chart_root, prompt, max_items=max_items)
