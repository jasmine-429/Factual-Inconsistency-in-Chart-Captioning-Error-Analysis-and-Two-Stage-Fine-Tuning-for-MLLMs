
import json

# 输入输出路径
input_path = "/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/merged_output_with_split.jsonl"
output_path = "/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/output_with_id.jsonl"

with open(input_path, "r") as infile, open(output_path, "w") as outfile:
    for line in infile:
        data = json.loads(line)

        model_name = data.get("model_name", "unknown_model")
        imgname = data.get("imgname", "unknown_image")

        new_id = f"{model_name}_{imgname}"

        # 创建新字典，确保 _id 插入在 model_name 后
        new_data = {}
        for key in data:
            new_data[key] = data[key]
            if key == "model_name":
                new_data["_id"] = new_id  # 插入在 model_name 之后

        outfile.write(json.dumps(new_data, ensure_ascii=False) + "\n")

print(f"✅ 生成成功，文件写入：{output_path}")
