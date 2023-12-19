import argparse
import json
import datetime
import os

'''
orig_data: {"query": "支付宝举报中心能受理哪些风险？", "answer": "支付宝官方举报中心能受理欺诈、被盗、内容违规等风险。"}
convert: {
    "id": "atech_qa-2", 
    "conversations": [
        {"from": "human", "value": "支付宝举报中心能受理哪些风险？"}, 
        {"from": "assistant", "value": "支付宝官方举报中心能受理欺诈、被盗、内容违规等风险。"}
        ]
    }
'''

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--orig_data",
    )
    parser.add_argument(
        "--write_data",
    )
    parser.add_argument(
        "--dataset_name",
    )
    args = parser.parse_args()
    f_write = open(args.write_data,"w")
    with open(args.orig_data) as f:
        lines = f.readlines()
        num_id = 1
        for line in lines:
            data = json.loads(line)
            # human_input = "回答问题：{}, 根据给定的文档信息：{}".format(data['query'], data['doc'])
            human_input = data['input']
            human_input_str = human_input["prompt"] + '\\n' + human_input["context"] + '\\n' + human_input["current_turn"]
            assistant = str(data["output"])
            conversations = [{"from": "human", "value": human_input_str},{"from": "assistant", "value": assistant}]
            # conversations = [{"from": "human", "value": data['input']},{"from": "assistant", "value": data['target']}]
            uniq_id = data['id'] if "id" in data else args.dataset_name+"-"+str(num_id)
            item = {"id":uniq_id, "conversations": conversations}
            f_write.write(json.dumps(item, ensure_ascii=False)+"\n")
            num_id += 1
    f_write.close()


if __name__ == "__main__":
    main()

