import torch
import os
from safetensors import safe_open

# 路径设置
checkpoint_path = "/data/jguo376/project/model/TinyChart/checkpoints/chart_entail/checkpoint-800"
model_path = "/data/jguo376/pretrained_models/TinyChart-3B-768"

print("Checking LoRA adapter...")
adapter_path = os.path.join(checkpoint_path, 'adapter_model.safetensors')

# 加载LoRA权重并查看
adapter_weights = {}
with safe_open(adapter_path, framework="pt", device="cpu") as f:
    for key in f.keys():
        adapter_weights[key] = f.get_tensor(key)

print(f"LoRA adapter has {len(adapter_weights)} parameters")
mm_keys = [k for k in adapter_weights.keys() if 'mm_projector' in k]
print(f"Found {len(mm_keys)} mm_projector keys in LoRA adapter")

if mm_keys:
    for key in mm_keys:
        print(f"  {key}")

# 检查原始模型目录
print(f"\nChecking original model directory: {model_path}")
model_files = os.listdir(model_path)
print("Original model files:")
for f in model_files:
    print(f"  {f}")

# 尝试从原始模型中提取mm_projector
possible_model_files = [
    'pytorch_model.bin',
    'pytorch_model-00001-of-00003.bin',  # 分片模型的第一个文件
    'pytorch_model-00002-of-00003.bin',
    'pytorch_model-00003-of-00003.bin',
    'model.safetensors'
]

mm_projector_weights = {}
found_mm_projector = False

for model_file in possible_model_files:
    model_file_path = os.path.join(model_path, model_file)
    if os.path.exists(model_file_path):
        print(f"Loading {model_file}...")
        
        try:
            if model_file.endswith('.safetensors'):
                with safe_open(model_file_path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        if 'mm_projector' in key:
                            clean_key = key.replace('model.', '')
                            mm_projector_weights[clean_key] = f.get_tensor(key)
                            found_mm_projector = True
            else:
                checkpoint = torch.load(model_file_path, map_location='cpu')
                for key, value in checkpoint.items():
                    if 'mm_projector' in key:
                        clean_key = key.replace('model.', '')
                        mm_projector_weights[clean_key] = value
                        found_mm_projector = True
                        
            if found_mm_projector:
                print(f"Found mm_projector weights in {model_file}")
                break
                
        except Exception as e:
            print(f"Error loading {model_file}: {e}")
            continue

if found_mm_projector:
    print(f"Extracted {len(mm_projector_weights)} mm_projector parameters:")
    for key in mm_projector_weights.keys():
        print(f"  {key}: {mm_projector_weights[key].shape}")
    
    # 保存mm_projector.bin
    output_path = os.path.join(model_path, 'mm_projector.bin')
    torch.save(mm_projector_weights, output_path)
    print(f"mm_projector.bin saved to {output_path}")
else:
    print("No mm_projector weights found in any model files!")
    print("This might indicate the model wasn't properly trained with vision components.")