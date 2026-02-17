import json

input_path = "/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/all_type/caption/deep_analysis/caption_all/merged_output.json"

with open(input_path, "r", encoding="utf-8") as f:
    content = f.read()

# 尝试修复文件开头或结尾格式问题
if content.strip().startswith('[') and content.strip().endswith(']'):
    content = content.strip()[1:-1].strip()  # 去掉开头结尾的中括号
else:
    print("⚠️ 文件可能不是合法的 JSON 数组开头或结尾")

# 按 } 拆分每个 item
items_raw = content.split('},')
errors = []

for i, chunk in enumerate(items_raw):
    if i < len(items_raw) - 1:
        chunk += '}'  # 补上 }

    try:
        json.loads(chunk)
    except Exception as e:
        print(f"\n❌ 第 {i+1} 条 JSON 出错")
        print("🧨 错误类型:", str(e))
        print("📍 内容预览:", chunk[:300])
        errors.append(i)

print(f"\n✅ 共检查 {len(items_raw)} 条，出错 {len(errors)} 条。")
