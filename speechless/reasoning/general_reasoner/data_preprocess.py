import re
import os
import datasets
from datasets import Dataset
import json
from verl.utils.hdfs_io import copy, makedirs
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', default='TIGER-Lab/WebInstruct-verified')
    parser.add_argument('--local-dir', default='~/webinstruct-verified')

    args = parser.parse_args()

    data_source = 'general-reasoner'

    dataset = datasets.load_dataset(args.dataset_name)
    
    train_dataset = dataset['train']
    test_dataset = dataset['test'].select(range(100))

    instruction_following = "Please reason step by step, and put your final answer within \\boxed{}."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            question_raw = example.pop('question')
            question = question_raw + ' ' + instruction_following
            answer = example.pop('answer')
            question_level = example.pop('difficulty')
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "reasoning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'answer': answer,
                    "question": question_raw,
                    'level': question_level
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))