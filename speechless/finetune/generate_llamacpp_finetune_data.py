#!/usr/bin/env python
import os, json
from tqdm import tqdm


PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:\n"
    ),
}

def generate_alpaca_gpt4_data(args):
    data_file = "/opt/local/datasets/alpaca_gpt4/alpaca_gpt4_data.json"
    json_data = json.load(open(data_file, "r"))
    
    with open(args.output_file, "w") as fd:
        for data in tqdm(json_data, ncols=100):
            instruction = data['instruction']
            input = data['input']
            output = data['output']
            if input:
                prompt = PROMPT_DICT["prompt_input"].format(instruction=instruction, input=input)
            else:
                prompt = PROMPT_DICT["prompt_no_input"].format(instruction=instruction)
            line = f"<s>{prompt}{output}\n"
            fd.write(line + "\n")
    print(f"Saved {len(json_data)} lines to {args.output_file}")

def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", type=str, default="llamacpp_finetune_data.txt", help="Path to the output file.")
    return parser.parse_args()

def main():
    args = get_args()
    generate_alpaca_gpt4_data(args)

if __name__ == "__main__":
    main()
