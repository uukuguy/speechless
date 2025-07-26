import matplotlib.pyplot as plt
import json
from e3.trainer.reward_fn import compute_score
from collections import defaultdict
from transformers import AutoTokenizer
from tqdm import tqdm


def get_acc_and_length(file, tokenizer):
    data = []
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    
    acc_list = defaultdict(list)
    count = defaultdict(int)
    length = defaultdict(int)
    for item in tqdm(data, desc="evaluating", unit="item"):
        acc=[]
        for res in item["test_outputs"]:
            if compute_score(res, item["answer"])["acc"] == 1:
                acc.append(1)
            else:
                acc.append(0)
            length[item["source"]] += len(tokenizer.encode(res))
            count[item["source"]] += 1
        acc_list[item["source"]].append(acc)
    for key in count:
        length[key] = length[key]/count[key]
    return {"acc":acc_list,"length":length}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_file", type=str, default='./.results/res.json')
    parser.add_argument("--output_file", type=str, default='./.metrics/res.json')
    parser.add_argument("--model_name", type=str, default='../checkpoints/model/')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    res= get_acc_and_length(args.result_file,tokenizer)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(res,f,ensure_ascii=False)
