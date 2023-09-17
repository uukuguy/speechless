#!/usr/bin/env python
import os, json
from tqdm import tqdm
import datasets
from transformers import (
    AutoTokenizer,
)


def prepare_data(model_name_or_path, model_max_len):
    # Tokenizer
    tokenizer_kwargs = {
        "cache_dir": None,
        "padding_side": "left",
        "use_fast": False,
    }
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **tokenizer_kwargs)

    ### jondurbin/airoboros-2.2
    experts = {
      "qa": [
        "quiz",
        "multiple_choice",
        "contextual",
        "counterfactual_contextual"
      ],
    #   "creative": [
    #     "card",
    #     "writing",
    #     "experience",
    #     "song",
    #     "roleplay",
    #     "gtkm",
    #     "rp",
    #     "detailed_writing",
    #     "joke"
    #   ],
      "code": [
        "coding"
      ],
      "reasoning": [
        "cot",
        "theory_of_mind",
        "riddle",
        "orca"
      ],
      "function": [
        "agent",
        "plan"
      ],
    #   "general": [
    #     "wordgame",
    #     "trivia",
    #     "general"
    #   ]
    }
    selected_categories = []
    for k, cats in experts.items():
        selected_categories.extend(cats)

    print(f"Loading jondurbin/airoboros-2.2 ...")
    airoboros_dataset = datasets.load_dataset("json", data_files="/opt/local/datasets/jondurbin/airoboros-2.2/instructions-clean.jsonl")['train']
    print(f"Loaded {len(airoboros_dataset)} samples from jondurbin/airoboros-2.2")
    total_samples = len(airoboros_dataset)
    airoboros_dataset = airoboros_dataset.filter(lambda x: x['category'] in selected_categories)
    remaining_samples = len(airoboros_dataset)
    print(f"Filter airoboros dataset from {total_samples} to {remaining_samples}. {remaining_samples/total_samples * 100:.2f}%")

    # Open-Orca/OpenOrca
    print(f"Loading Open-Orca/OpenOrca ...")
    def load_orca_dataset(train_data_path: str):
        ds = datasets.load_dataset(train_data_path)
        ds = ds['train']
        total_samples = len(ds)
        print(f"Loaded {len(ds)} samples from Open-Orca/OpenOrca")
        ds = ds.filter(lambda x: x['id'].startswith('cot.'))
        remaining_samples = len(ds)
        print(f"Filter orca dataset from {total_samples} to {remaining_samples}. {remaining_samples/total_samples * 100:.2f}%")
        ds = ds.remove_columns(['id'])
        ds = ds.rename_column('system_prompt', 'system')
        ds = ds.rename_column('question', 'instruction')
        # ds = ds.rename_column('response', 'response')
        ds = ds.add_column('category', ['cot']  * len(ds)) 
        ds = ds.add_column("skip_prompt_formatting", [False] * len(ds))
        return ds

    orca_dataset = load_orca_dataset("/opt/local/datasets/OpenOrca")

    # garage-bAInd/Open-Platypus
    print(f"Loading garage-bAInd/Open-Platypus ...")
    def load_platypus_dataset(train_data_path: str):
        ds = datasets.load_dataset(train_data_path)
        ds = ds['train']
        total_samples = len(ds)
        print(f"Loaded {len(ds)} samples from garage-bAInd/Open-Platypus")
        ds = ds.remove_columns(['input'])
        ds = ds.add_column('system', ['']  * len(ds)) 
        # ds = ds.rename_column('instruction', 'instruction')
        ds = ds.rename_column('output', 'response')
        ds = ds.add_column('category', ['platypus']  * len(ds)) 
        ds = ds.add_column("skip_prompt_formatting", [False] * len(ds))
        return ds

    platypus_dataset = load_platypus_dataset("/opt/local/datasets/garage-bAInd/Open-Platypus")

    dataset = datasets.concatenate_datasets([airoboros_dataset, orca_dataset, platypus_dataset])

    print(f"Merging all datasets ...")
    def _get_data_length(item):
        prompt = f"{tokenizer.bos_token}{item['instruction']}{item['response']}{tokenizer.eos_token}"
        return len(
            tokenizer(
                prompt,
                max_length=model_max_len + 1,
                truncation=True,
                add_special_tokens=False
            ).input_ids
        )

    # total_samples = len(dataset)
    # dataset = dataset.filter(lambda x: _get_data_length(x) < model_max_len - 20)
    # remaining_samples = len(dataset)
    # print(f"Filter dataset from {total_samples} to {remaining_samples}. {remaining_samples/total_samples * 100:.2f}%")

    dataset_file = "/opt/local/datasets/Speechless/airoboros-orca-platypus-instructions.jsonl"
    dataset.to_json(dataset_file, orient="records", lines=True, index=False)
    print(f"Saved {len(dataset)} samples to {dataset_file}")

def main():
    model_name_or_path="/opt/local/llm_models/huggingface.co/codellama/CodeLlama-13b-hf"
    model_max_len = 4096
    prepare_data(model_name_or_path=model_name_or_path, model_max_len=model_max_len)
  
if __name__ == '__main__':
    main()