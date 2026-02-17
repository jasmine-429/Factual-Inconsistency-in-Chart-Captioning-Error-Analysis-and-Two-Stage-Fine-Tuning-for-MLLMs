# merge_two_adapters_into_one.py
import os
import json
import shutil
import argparse
import torch

def load_adapter_bin(path):
    bin_path = os.path.join(path, "adapter_model.bin")
    if not os.path.exists(bin_path):
        raise FileNotFoundError(f"找不到 {bin_path}")
    return torch.load(bin_path, map_location="cpu")

def load_adapter_config(path):
    cfg_path = os.path.join(path, "adapter_config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"找不到 {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)

def check_compat(cfg1, cfg2, keys=("peft_type","task_type","r","lora_alpha","lora_dropout","target_modules")):
    for k in keys:
        if cfg1.get(k) != cfg2.get(k):
            raise ValueError(f"LoRA 配置不一致: key={k}, v1={cfg1.get(k)}, v2={cfg2.get(k)}")

def merge_state_dicts(sd1, sd2, w1=1.0, w2=1.0):
    merged = {}
    # 以 sd1 的键为基准，若 sd2 有同名键则合并，否则保留 sd1
    keys = set(sd1.keys()) | set(sd2.keys())
    for k in keys:
        v1 = sd1.get(k, None)
        v2 = sd2.get(k, None)
        if isinstance(v1, torch.Tensor) and isinstance(v2, torch.Tensor):
            # 简单线性合并：v = w1*v1 + w2*v2
            merged[k] = w1 * v1 + w2 * v2
        elif isinstance(v1, torch.Tensor) and v2 is None:
            merged[k] = v1.clone()
        elif v1 is None and isinstance(v2, torch.Tensor):
            merged[k] = v2.clone()
        else:
            # 既不是 Tensor 就跳过（常见是元数据，不影响）
            pass
    return merged

def main(args):
    lora1_dir = args.lora1
    lora2_dir = args.lora2
    out_dir   = args.out
    w1        = args.w1
    w2        = args.w2
    base_name = args.base  # 可选，写入 adapter_config.json 里

    os.makedirs(out_dir, exist_ok=True)

    print("📥 读取 LoRA 1/2 的权重与配置...")
    sd1 = load_adapter_bin(lora1_dir)
    sd2 = load_adapter_bin(lora2_dir)
    cfg1 = load_adapter_config(lora1_dir)
    cfg2 = load_adapter_config(lora2_dir)

    print("🔎 校验关键 LoRA 配置是否一致（r/alpha/target_modules 等）...")
    check_compat(cfg1, cfg2)

    print(f"🧮 合并权重：merged = {w1} * lora1 + {w2} * lora2")
    merged_sd = merge_state_dicts(sd1, sd2, w1=w1, w2=w2)

    # 写出合并后的 adapter_model.bin
    out_bin = os.path.join(out_dir, "adapter_model.bin")
    torch.save(merged_sd, out_bin)
    print(f"💾 已保存: {out_bin}")

    # 生成合并后的 adapter_config.json（以 cfg1 为基准，可选写入 base_model_name_or_path）
    merged_cfg = cfg1.copy()
    if args.base:
        merged_cfg["base_model_name_or_path"] = base_name
    out_cfg = os.path.join(out_dir, "adapter_config.json")
    with open(out_cfg, "w", encoding="utf-8") as f:
        json.dump(merged_cfg, f, indent=2, ensure_ascii=False)
    print(f"💾 已保存: {out_cfg}")

    print("✅ 完成！现在你可以像普通 LoRA 一样加载这个合并后的适配器。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora1", required=True, help="LoRA 1 目录（含 adapter_model.bin / adapter_config.json）")
    parser.add_argument("--lora2", required=True, help="LoRA 2 目录")
    parser.add_argument("--out",   required=True, help="输出目录（将生成合并后的 LoRA）")
    parser.add_argument("--w1",    type=float, default=1.0, help="LoRA1 权重系数")
    parser.add_argument("--w2",    type=float, default=1.0, help="LoRA2 权重系数")
    parser.add_argument("--base",  type=str, default="",     help="可选：写入 base_model_name_or_path 字段")
    args = parser.parse_args()
    main(args)
