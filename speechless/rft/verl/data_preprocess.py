import re
import os
from datasets import load_dataset
import json
import argparse


if __name__ == '__main__':
    data_source = 'difc2025-round-a'

    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset-name', default='../../difc2025_round_a_rft_train_data_all.jsonl')
    parser.add_argument('--dataset-name', default='../../difc2025_round_a_rft_train_data_all-8k.jsonl')
    parser.add_argument('--local-dir', default=f'./data/{data_source}')
    parser.add_argument('--val-size', default=20)

    args = parser.parse_args()
    val_size = args.val_size
    dataset_name = args.dataset_name

    if dataset_name.endswith('.jsonl'):
        all_dataset = load_dataset('json', data_files=dataset_name, split='train')
    else:
        all_dataset = load_dataset(dataset_name, split='train')
    print(f"{all_dataset=}")

    train_test = all_dataset.train_test_split(test_size=val_size, seed=10042)
    train_dataset = train_test['train']
    val_dataset = train_test['test']

    print(f"{train_dataset=}")
    print(f"{val_dataset=}")

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            instruction = example.pop('instruction')
            instruction += "\n\n/no_think"
            response = example.pop('response')
            category = example.pop('category')
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": instruction,
                }],
                "ability": "reasoning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": response
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'response': response,
                    "instruction": instruction,
                    'category': category,
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    val_dataset = val_dataset.map(function=make_map_fn('val'), with_indices=True)

    local_dir = args.local_dir
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    val_dataset.to_parquet(os.path.join(local_dir, 'val.parquet'))