import json


input_file = "data/qwen_data.json"
output_file = "data/qwen_data_2.json"


data = json.load(open(input_file, "r"))

image_tag = "<image>\n<image>\n"

for item in data:
    for conversation in item.get("conversations", []):
        value = conversation.get("value", "")
        
        # 如果结尾是 image_tag，则进行替换
        if value.endswith(image_tag):
            # 去掉末尾的 <image>\n<image>\n，然后将它拼接到最前面
            trimmed_value = value[:-len(image_tag)]
            conversation["value"] = image_tag + trimmed_value

with open(output_file, "w") as f:
    json.dump(data, f)

print('done')