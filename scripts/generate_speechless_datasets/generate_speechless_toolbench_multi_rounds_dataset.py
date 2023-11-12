#!/usr/bin/env python
import os, json
from tqdm import tqdm
import datasets
from transformers import (
    AutoTokenizer,
)

def load_toolbench_dataset():
    dataset_name_or_path = "/opt/local/datasets/toolllama/ToolBench"
    dataset = datasets.load_dataset(dataset_name_or_path)
    # dataset = datasets.concatenate_datasets(dataset.values())
    dataset = dataset['train']

    def _format(sample):
        convs = sample['conversations']  
        conversations = []
        for round in convs:
            conversations.append({
                "from": round["from"],
                "value": round["value"],
            })
        return {
            'category': "toolbench",
            'prompt_type': "tool-llama-multi-rounds",
            "system_prompt": "",
            'conversations': conversations,
        }

    dataset = dataset.map(_format)

    dataset = dataset.remove_columns(['id'])
    return dataset

# total_samples = len(dataset)
# dataset = dataset.filter(_filter_by_token_length)
# remaining_samples = len(dataset)
# print(f"Filter dataset from {total_samples} to {remaining_samples}. {remaining_samples/total_samples * 100:.2f}%")

# print(f"Convert to sharegpt format ...")
# dataset = dataset.map(lambda x: {
#     'category': x['category'],
#     'prompt_type': "alpaca",
#     'system_prompt': "", #x['system'],
#     'dialog': [
#             {
#                 'from': 'human',
#                 'value': x['instruction'].replace("PLAINFORMAT", ""),  
#             },
#             {
#                 'from': 'assistant',
#                 'value': x['response'],  
#             },
#     ]
# })


def prepare_data(model_name_or_path, model_max_len):
    # Tokenizer
    tokenizer_kwargs = {
        "cache_dir": None,
        "padding_side": "left",
        "use_fast": False,
    }
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **tokenizer_kwargs)

    dataset_file = "/opt/local/datasets/speechless_data/speechless-toolbench-multi-rounds-8k.jsonl"
    toolbench_dataset = load_toolbench_dataset()
    dataset = toolbench_dataset

    # Remove unused columns.
    dataset = dataset.remove_columns(
        # FIXME
        [col for col in dataset.column_names if col not in ['conversations', 'system_prompt', 'category', 'prompt_type']]
    )
    print(dataset[0])

    # dataset = datasets.concatenate_datasets([dataset, agent_instruct_dataset])

    def _get_data_length(item, tokenizer):
        # prompt = f"{tokenizer.bos_token}{item['instruction']}{item['response']}{tokenizer.eos_token}"
        prompt = f"{tokenizer.bos_token}{item['conversations']}{tokenizer.eos_token}"
        return len(
            tokenizer(
                prompt,
                # max_length=model_max_len + 1,
                truncation=False,
                add_special_tokens=False
            ).input_ids
        )

    def _filter_by_token_length(item):
        tok_len =  _get_data_length(item, tokenizer=tokenizer)
        return tok_len >= 512 and tok_len <= model_max_len - 50

    total_samples = len(dataset)
    dataset = dataset.filter(_filter_by_token_length)
    remaining_samples = len(dataset)
    print(f"Filter dataset from {total_samples} to {remaining_samples}. {remaining_samples/total_samples * 100:.2f}%")


    dataset.to_json(dataset_file, orient="records", lines=True, index=False)
    print(f"Saved {len(dataset)} samples to {dataset_file}")

def main():
    model_name_or_path="/opt/local/llm_models/huggingface.co/llm_agents/tora-code-7b-v1.0"
    model_max_len = 1024 * 8
    prepare_data(model_name_or_path=model_name_or_path, model_max_len=model_max_len)
  
if __name__ == '__main__':
    main()