import sys
sys.path.append("/data/jguo376/project/model/TinyChart")

import os
import torch
from tinychart.model.builder import load_pretrained_model

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

# 路径配置
base_model_path = "/data/jguo376/pretrained_models/TinyChart-3B-768"
lora_path = "/data/jguo376/project/model/TinyChart/checkpoints/chart_entail/checkpoint-800"

def load_lora_model():
    print("=== 加载 TinyChart LoRA 模型 ===")
    
    try:
        # 加载基础模型
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=base_model_path,
            model_base=None,
            model_name="tinychart",
            device="cuda"
        )
        
        # 手动加载 LoRA
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()
        
        # 关键修复：统一数据类型
        model = model.half()
        
        print("✅ LoRA 加载成功！")
        return tokenizer, model, image_processor, context_len
        
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return None, None, None, None

def test_model_inference(tokenizer, model, image_processor):
    print("\n=== 测试模型推理 ===")
    
    try:
        # 测试文本
        test_prompt = "Does the image entails this statement: \"Test statement\"?"
        
        # 创建测试图像
        from PIL import Image
        import numpy as np
        test_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        
        # 处理输入
        from tinychart.conversation import conv_templates
        conv = conv_templates["phi"].copy()
        conv.append_message(conv.roles[0], test_prompt)
        conv.append_message(conv.roles[1], None)
        text = conv.get_prompt()
        
        inputs = tokenizer([text], return_tensors="pt").to("cuda")
        # 关键修复：确保图像张量数据类型一致
        image_tensor = image_processor.preprocess(test_image, return_tensors="pt")["pixel_values"][0].unsqueeze(0)
        image_tensor = image_tensor.to("cuda").to(model.dtype)
        
        # 推理
        with torch.no_grad():
            outputs = model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                images=image_tensor,
                return_dict=True,
            )
        
        print(f"✅ 推理测试成功! 输出形状: {outputs.logits.shape}")
        return True
        
    except Exception as e:
        print(f"❌ 推理测试失败: {e}")
        return False

if __name__ == "__main__":
    # 加载模型
    tokenizer, model, image_processor, context_len = load_lora_model()
    
    if model is not None:
        # 测试推理
        success = test_model_inference(tokenizer, model, image_processor)
        
        if success:
            print("\n🎉 所有测试通过！模型可以用于评估。")
        else:
            print("\n⚠️ 模型加载成功但推理测试失败。")
    else:
        print("\n❌ 模型加载失败。")