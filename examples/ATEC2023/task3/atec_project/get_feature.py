# 读取特征
# 制作新的数据
# 保存数据
import os
import json


def read_jsonl_file(file_path):
    """
    读取JSON Lines文件，假设每行都有'id', 'input' 和 'output' 属性。

    参数:
        file_path (str): JSON Lines文件的路径。

    返回:
        records (list): 包含所有记录的列表，每个记录都是一个字典。
    """
    records = []

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    record = json.loads(line.strip())  # 去除可能的空白字符后解析JSON
                    if 'id' in record and 'input' in record and 'output' in record:
                        records.append(record)
                    else:
                        print(f"警告: 一行数据缺少 'id', 'input' 或 'output' 键: {line}")
                except json.JSONDecodeError:
                    print(f"无法解析的JSON行: {line}")
        return records
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到。")
    except Exception as e:
        print(f"读取文件 {file_path} 时发生错误：{e}")


def add_feature(input):
    prefix = "请你帮我判断如下内容是人类撰写的还是机器生成的，内容如下《%s》，注意回复为该新闻是由人类撰写的或该新闻是由机器合成的。" % (input)
    return prefix


if __name__ == "__main__":
    # 用法示例：
    train_jsonl_file = "/home/admin/workspace/job/input/train.jsonl"
    dev_jsonl_file = "/home/admin/workspace/job/input/dev.jsonl"

    # 构建特征数据
    os.makedirs("./data", exist_ok=True)
    train_json_file = "./data/train.json"
    dev_json_file = "./data/dev.json"

    #
    train_jsonl_content = read_jsonl_file(train_jsonl_file)
    for content in train_jsonl_content:
        content['input'] = add_feature(content['input'])

    with open(train_json_file, 'w', encoding='utf-8') as file:
        json.dump(train_jsonl_content, file, ensure_ascii=False, indent=4)

    print("train data done!")

    dev_jsonl_content = read_jsonl_file(dev_jsonl_file)
    for content in dev_jsonl_content:
        content['input'] = add_feature(content['input'])

    with open(dev_json_file, 'w', encoding='utf-8') as file:
        json.dump(dev_jsonl_content, file, ensure_ascii=False, indent=4)

    print("val data done!")
