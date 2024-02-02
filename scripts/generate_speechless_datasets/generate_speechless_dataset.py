#!/usr/bin/env python
import os, json
from tqdm import tqdm
import datasets
from transformers import (
    AutoTokenizer,
)

def load_nl2sql_spider_birdsql_dataset(train_data_path:str):
    print(f"Loading NL2SQL-Spider-BirdSQL ...")
    ds = datasets.load_dataset("json", data_files=train_data_path)['train']
    instructions = ds['instruction']
    inputs = ds['input']
    responses = ds['response']

    instructions = [ f"{a}{b}" for a, b in zip(instructions, inputs)]

    ds = datasets.Dataset.from_dict({
        'instruction': instructions,
        'response': responses,
        'category': ['nlsql-spider-birdsql'] * len(instructions)
    })

    return ds

def prepare_data(model_name_or_path, model_max_len, model_min_len):
    # Tokenizer
    tokenizer_kwargs = {
        "cache_dir": None,
        "padding_side": "left",
        "use_fast": False,
    }
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **tokenizer_kwargs)

    dataset_file = "/opt/local/datasets/speechless_data/speechless-nl2sql-18087.jsonl"

    nl2sql_spider_birdsql_dataset = load_nl2sql_spider_birdsql_dataset("/opt/local/datasets/speechless-nl2sql-spider-birdsql-18087.jsonl")

    dataset = nl2sql_spider_birdsql_dataset

    def _get_data_length(item):
        # prompt = f"{tokenizer.bos_token}{item['instruction']}{item['response']}{tokenizer.eos_token}"
        prompt = f"{tokenizer.bos_token}{item['conversations'][0]['value']}{item['conversations'][1]['value']}{tokenizer.eos_token}"
        return len(
            tokenizer(
                prompt,
                # max_length=model_max_len + 1,
                truncation=False,
                add_special_tokens=False
            ).input_ids
        )

    def _filter_by_token_length(item):
        tok_len = _get_data_length(item)
        return tok_len >= model_min_len and tok_len <= model_max_len - 100

    total_samples = len(dataset)
    dataset = dataset.filter(_filter_by_token_length)
    remaining_samples = len(dataset)
    print(f"Filter dataset from {total_samples} to {remaining_samples}. {remaining_samples/total_samples * 100:.2f}%")

    print(f"Convert to sharegpt format ...")
    dataset = dataset.map(
        lambda x: {
            'category':
            x['category'],
            'conversations': [
                {
                    'from': 'human',
                    'value': x['instruction'].replace("PLAINFORMAT", ""),
                },
                {
                    'from': 'assistant',
                    'value': x['response'],
                },
            ]
        }
    )
    # Remove unused columns.
    dataset = dataset.remove_columns(
        # FIXME
        [
            col for col in dataset.column_names
            if col not in ['conversations', 'category']
        ]
    )

    dataset.to_json(dataset_file, orient="records", lines=True, index=False)
    print(f"Saved {len(dataset)} samples to {dataset_file}")

def main():
    model_name_or_path = "/opt/local/llm_models/huggingface.co/meta-llama/Llama-2-7b-hf"
    model_max_len = 1024 * 8
    model_min_len = 128
    prepare_data(model_name_or_path=model_name_or_path, model_max_len=model_max_len, model_min_len=model_min_len)


if __name__ == '__main__':
    main()