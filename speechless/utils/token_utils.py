"""
This module contains functions to count the number of tokens used in a prompt. It uses the tiktoken library to count the number of tokens used in a prompt. It also provides a function to count the number of tokens used in a prompt.
"""
import json
from typing import List, Dict
import tiktoken


def titoken_count_tokens(
    model_name: str = "gpt-3.5-turbo-0613", messages: List[Dict[str, str]] = [], prompt: str = ""
) -> int:
    """
    This function counts the number of tokens used in a prompt.
    model_name: the model used to generate the prompt. can be one of the following: gpt-3.5-turbo-0613, gpt-4-0613, text-davinci-003
    messages: (only for OpenAI chat models) a list of messages to be used as a prompt. Each message is a dict with two keys: role and content
    prompt: (only for text-davinci-003 model) a string to be used as a prompt
    Returns the number of tokens used in the prompt as an integer.
    """
    gpt_models = ['gpt-3.5-turbo-0613', 'gpt-4-0613', 'text-davinci-003']
    if model_name in gpt_models:
        tokenizer = tiktoken.encoding_for_model(model_name)
    else:
        tokenizer = tiktoken.encoding_for_model('gpt-3.5-turbo-0613')

    num_tokens = 0
    if messages:
        for message in messages:
            for _, value in message.items():
                num_tokens += len(tokenizer.encode(value))
    else:
        num_tokens = len(tokenizer.encode(prompt))

    return num_tokens


def get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument(
        "--speechless_data_file",
        type=str,
        default="/opt/local/datasets/speechless_data/speechless-thoughts-252k.jsonl",
        help="speechless data file"
    )
    parser.add_argument("--prompt_file", type=str, help="prompt file")

    args = parser.parse_args()
    return args


def count_tokens_prompt_file(prompt_file):
    lines = open(prompt_file).readlines()

    num_tokens = 0
    from tqdm import tqdm
    pbar = tqdm(lines, ncols=100)
    for idx, line in enumerate(pbar):
        prompt = line.strip()
        num_tokens += titoken_count_tokens(prompt=prompt)
        tokens_per_line = num_tokens / (idx+1)
        pbar.set_postfix({
            "tokens": f"{num_tokens:.2e}",
            "tokens/line": f"{tokens_per_line:.2f}"
        })

    print(f"Total tokens: {num_tokens:.2e}, tokens per line: {tokens_per_line:.2f}")
    # pretraining_data_1024.txt 
    # 17.7K lines, 111MB
    # Total tokens: 5.42e+07, tokens per line: 3066.39
    # pretraining_data.txt
    # 8.7K lines, 111MB
    # Total tokens: 5.41e+07, tokens per line: 6214.97 
    # pretraining_data_256.txt
    # 39.8K lines, 112MB
    # Total tokens: 5.43e+07, tokens per line: 1365.09
    

def count_tokens_speechless_data_file(speechless_data_file):
    lines = open(speechless_data_file).readlines()

    num_tokens = 0
    from tqdm import tqdm
    pbar = tqdm(lines, ncols=100)
    for idx, line in enumerate(pbar):
        data = json.loads(line.strip())
        messages = data['conversations']
        num_tokens += titoken_count_tokens(messages=messages)
        tokens_per_line = num_tokens / (idx+1)
        pbar.set_postfix({
            "tokens": f"{num_tokens:.2e}",
            "tokens/line": f"{tokens_per_line:.2f}"
        })

    print(f"Total tokens: {num_tokens:.2e}, tokens per line: {tokens_per_line:.2f}")
    # speechless-thoughts-252k.jsonl
    # 25.2K lines, 456MB
    # Total tokens: 1.03e+08, tokens per line: 409.38

    # speechless-herm-code-986k.jsonl
    # 98.6K lines, 1.7GB
    # Total tokens: 3.98e+08, tokens per line: 403.94

def main():
    args = get_args()
    if args.prompt_file:
        count_tokens_prompt_file(args.prompt_file)
    elif args.speechless_data_file:
        count_tokens_speechless_data_file(args.speechless_data_file)
    else:
        raise ValueError("prompt_file or speechless_data_file must be provided")

if __name__ == '__main__':
    main()
