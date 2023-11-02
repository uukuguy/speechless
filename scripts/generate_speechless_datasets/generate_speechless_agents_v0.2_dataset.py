#!/usr/bin/env python
import os, json
from tqdm import tqdm
import datasets
from transformers import (
    AutoTokenizer,
)

def load_agent_instruct_dataset():
    dataset_name_or_path = "/opt/local/datasets/huggingface.co/THUDM/AgentInstruct"
    dataset = datasets.load_dataset(dataset_name_or_path)
    dataset = datasets.concatenate_datasets(dataset.values())
    def _format(sample):
        convs = sample['conversations']  
        dialog = []
        for i in range(len(convs) // 2):
            human = convs[i*2]
            gpt = convs[i*2+1]
            human = {'from': human['from'], 'value': human['value']}
            gpt = {'from': gpt['from'], 'value': gpt['value']}
            dialog.append(human)
            dialog.append(gpt)
        return {
            'category': "AgentInstruct",
            'system_prompt': "You are a helpful, respectful and honest assistant.",
            'dialog': dialog,
        }

    dataset = dataset.map(_format)

    dataset = dataset.remove_columns(['conversations', 'id'])
    # dataset = dataset.add_column('category', ['AgentInstruct']  * len(dataset)) 
    # dataset = dataset.add_column('system_prompt', ["You are a helpful, respectful and honest assistant.\n"] * len(dataset))

    return dataset

# ---------- jondurbin/airoboros-2.2.1 ----------
def load_airoboros_dataset():
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

    print(f"Loading jondurbin/airoboros-2.2.1 ...")
    airoboros_dataset = datasets.load_dataset("json", data_files="/opt/local/datasets/jondurbin/airoboros-2.2.1/instructions.jsonl")['train']
    print(f"Loaded {len(airoboros_dataset)} samples from jondurbin/airoboros-2.2.1")
    total_samples = len(airoboros_dataset)
    airoboros_dataset = airoboros_dataset.filter(lambda x: x['category'] in selected_categories)
    remaining_samples = len(airoboros_dataset)
    print(f"Filter airoboros dataset from {total_samples} to {remaining_samples}. {remaining_samples/total_samples * 100:.2f}%")

    return airoboros_dataset

# ---------- jondurbin/airoboros-2.2 ----------
def load_airoboros_22_dataset():
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
    airoboros_dataset = datasets.load_dataset("json", data_files="/opt/local/datasets/jondurbin/airoboros-2.2/instructions.jsonl")['train']
    print(f"Loaded {len(airoboros_dataset)} samples from jondurbin/airoboros-2.2")
    total_samples = len(airoboros_dataset)
    airoboros_dataset = airoboros_dataset.filter(lambda x: x['category'] in selected_categories)
    remaining_samples = len(airoboros_dataset)
    print(f"Filter airoboros dataset from {total_samples} to {remaining_samples}. {remaining_samples/total_samples * 100:.2f}%")

    return airoboros_dataset

# ---------- Open-Orca/OpenOrca ----------
def load_orca_dataset(train_data_path: str):
    print(f"Loading Open-Orca/OpenOrca ...")
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

# ---------- garage-bAInd/Open-Platypus ----------
def load_platypus_dataset(train_data_path: str):
    print(f"Loading garage-bAInd/Open-Platypus ...")
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

# ---------- WizardLM/WizardLM_evol_instruct_V2_196k ----------
def load_wizardlm_dataset(train_data_path: str):
    print(f"Loading WizardLM/WizardLM_evol_instruct_V2_196k ...")
    json_file = f"{train_data_path}/WizardLM_evol_instruct_V2_143k.json"
    ds = datasets.load_dataset("json", data_files=json_file)['train']  
    total_samples = len(ds)
    print(f"Loaded {len(ds)} samples from WizardLM/WizardLM_evol_instruct_V2_196k")

    def _filter_fn(samples):
        conversations = samples['conversations']

        outputs = {
            'instruction': [],
            # 'input': [],
            'response': [],
            'system': [],
            'skip_prompt_formatting': [False] * len(conversations),
            'category': ['wizardlm'] * len(conversations),
        }
        for i in range(len(conversations)):
            human, gpt = conversations[i]
            instruction = human['value']
            response = gpt['value']
            # if "```" in response:
            #   print(f"{response=}")
            outputs['instruction'].append(instruction)
            # outputs['input'].append("")
            outputs['response'].append(response)
            outputs['system'].append("")

        return outputs

    ds = ds.map(_filter_fn, batched=True, remove_columns=ds.column_names)
    print(f"Mapped {len(ds)} in dataset")
    print(f"column_names: {ds.column_names}, {len(ds)=}")
    print(f"{type(ds)=}, {type(ds[0])=}")

    ds = ds.filter(lambda x: "```" in x['response'])

    remaining_samples = len(ds)
    print(f"Filter wizardlm dataset from {total_samples} to {remaining_samples}. {remaining_samples/total_samples * 100:.2f}%")

    return ds

# ---------- TokenBender/python_evol_instruct_51k ----------
def load_tokenbender_dataset(train_data_path: str):
    print(f"Loading TokenBender/python_evol_instruct_51k ...")
    json_file = f"{train_data_path}/tokenbender_python_evol_instruct_51k.jsonl"
    ds = datasets.load_dataset("json", data_files=json_file)['train']
    total_samples = len(ds)
    print(f"Loaded {len(ds)} samples from TokenBender/python_evol_instruct_51k")
    ds = ds.filter(lambda x: "```" in x['response'])
    remaining_samples = len(ds)
    print(f"Filter orca dataset from {total_samples} to {remaining_samples}. {remaining_samples/total_samples * 100:.2f}%")
    return ds

# ---------- nampdn-ai/tiny-codes ----------
def load_tinycodes_dataset(train_data_path: str):
    """
    ['prompt', 'main_topic', 'subtopic', 'adjective', 'action_verb', 'scenario', 'target_audience', 
    'programming_language', 'common_sense_topic', 'idx', 'response']
    Python, Java, JavaScript, C++, Rust, Go, Bash, Julia, relation database and SQL
    """
    print(f"Loading nampdn-ai/tiny-codes ...")
    ds = datasets.load_dataset(train_data_path)['train']
    # Python, Java, JavaScript, C++, Rust, Go, Bash, Julia, C#, TypeScript
    # selected_langs = ['Python', 'Java', 'JavaScript', 'C++', 'Rust', 'Go', 'Bash', 'Julia', 'relation database and SQL']
    # v0.2
    # selected_langs = ['Python', 'Java', 'JavaScript', 'C++', 'Rust', 'Go', 'Bash', 'Julia', 'C#', 'TypeScript']
    # FIXME v0.3
    selected_langs = ['Python', 'Java', 'JavaScript', 'C++', 'Rust', 'Go', 'Bash', 'Julia', 'TypeScript']

    ds_list = []
    for lang in selected_langs:
        print(f"promgramming language: {lang}")
        lang_ds = ds.filter(lambda x: x['programming_language'] == lang)
        print(f"{len(lang_ds)} {lang} samples. Selecting 10000 samples ...")
        lang_ds = lang_ds.train_test_split(test_size=10000)['test']
        ds_list.append(lang_ds)

    ds = datasets.concatenate_datasets(ds_list)

    ds = ds.rename_column('prompt', 'instruction')        
    tinycodes_column_names = [c for c in ds.column_names if c not in ['instruction', 'response']]
    ds = ds.remove_columns(tinycodes_column_names)
    # ds = ds.add_column('input', ['']  * len(ds))
    ds = ds.add_column('system', ['']  * len(ds))
    ds = ds.add_column('skip_prompt_formatting', [False] * len(ds))
    ds = ds.add_column('category', ['tinycodes'] * len(ds))

    return ds


def prepare_data(model_name_or_path, model_max_len):
    # Tokenizer
    tokenizer_kwargs = {
        "cache_dir": None,
        "padding_side": "left",
        "use_fast": False,
    }
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **tokenizer_kwargs)

    # # speechless-reasoning-v0.jsonl
    # dataset_file = "/opt/local/datasets/Speechless/speechless-reasoning-v0.jsonl"
    # airoboros_dataset = load_airoboros_dataset()
    # orca_dataset = load_orca_dataset("/opt/local/datasets/OpenOrca")
    # platypus_dataset = load_platypus_dataset("/opt/local/datasets/garage-bAInd/Open-Platypus")
    # wizardlm_dataset = load_wizardlm_dataset("/opt/local/datasets/WizardLM/WizardLM_evol_instruct_V2_196k")
    # dataset = datasets.concatenate_datasets(
    #   [airoboros_dataset, orca_dataset, platypus_dataset, wizardlm_dataset])

    # # speechless-reasoning-v0.1.jsonl
    # dataset_file = "/opt/local/datasets/Speechless/speechless-reasoning-v0.1.jsonl"
    # airoboros_dataset = load_airoboros_dataset()
    # orca_dataset = load_orca_dataset("/opt/local/datasets/OpenOrca")
    # platypus_dataset = load_platypus_dataset("/opt/local/datasets/garage-bAInd/Open-Platypus")
    # wizardlm_dataset = load_wizardlm_dataset("/opt/local/datasets/WizardLM/WizardLM_evol_instruct_V2_196k")
    # tokenbender_dataset = load_tokenbender_dataset("/opt/local/datasets/TokenBender/python_evol_instruct_51k")
    # dataset = datasets.concatenate_datasets(
    #   [airoboros_dataset, orca_dataset, platypus_dataset, wizardlm_dataset, tokenbender_dataset])

    # # speechless-reasoning-v0.2.jsonl
    # dataset_file = "/opt/local/datasets/Speechless/speechless-reasoning-v0.2.jsonl"
    # airoboros_dataset = load_airoboros_dataset()
    # orca_dataset = load_orca_dataset("/opt/local/datasets/OpenOrca")
    # platypus_dataset = load_platypus_dataset("/opt/local/datasets/garage-bAInd/Open-Platypus")
    # wizardlm_dataset = load_wizardlm_dataset("/opt/local/datasets/WizardLM/WizardLM_evol_instruct_V2_196k")
    # tinycodes_dataset = load_tinycodes_dataset("/opt/local/datasets/code/nampdn-ai/tiny-codes")
    # tinycodes_dataset = tinycodes_dataset.cast(airoboros_dataset.features)
    # dataset = datasets.concatenate_datasets(
    #   [airoboros_dataset, orca_dataset, platypus_dataset, wizardlm_dataset, tinycodes_dataset])

    # # speechless-reasoning-v0.3.jsonl
    # dataset_file = "/opt/local/datasets/Speechless/speechless-reasoning-v0.3.jsonl"
    # airoboros_dataset = load_airoboros_dataset()
    # orca_dataset = load_orca_dataset("/opt/local/datasets/OpenOrca")
    # platypus_dataset = load_platypus_dataset("/opt/local/datasets/garage-bAInd/Open-Platypus")
    # wizardlm_dataset = load_wizardlm_dataset("/opt/local/datasets/WizardLM/WizardLM_evol_instruct_V2_196k")
    # tinycodes_dataset = load_tinycodes_dataset("/opt/local/datasets/code/nampdn-ai/tiny-codes")
    # # FIXME v0.3 Remove C# in tinycodes_dataset
    # tinycodes_dataset = tinycodes_dataset.cast(airoboros_dataset.features)
    # dataset = datasets.concatenate_datasets(
    #   [airoboros_dataset, orca_dataset, platypus_dataset, wizardlm_dataset, tinycodes_dataset])

    # # speechless-reasoning-v0.4.jsonl
    # dataset_file = "/opt/local/datasets/Speechless/speechless-reasoning-v0.4.jsonl"
    # # FIXME airoboros 2.2
    # airoboros_dataset = load_airoboros_22_dataset()
    # orca_dataset = load_orca_dataset("/opt/local/datasets/OpenOrca")
    # platypus_dataset = load_platypus_dataset("/opt/local/datasets/garage-bAInd/Open-Platypus")
    # wizardlm_dataset = load_wizardlm_dataset("/opt/local/datasets/WizardLM/WizardLM_evol_instruct_V2_196k")
    # # FIXME use tokenbender
    # tokenbender_dataset = load_tokenbender_dataset("/opt/local/datasets/TokenBender/python_evol_instruct_51k")
    # # FIXME not use tinycodes
    # # tinycodes_dataset = load_tinycodes_dataset("/opt/local/datasets/code/nampdn-ai/tiny-codes")
    # # tinycodes_dataset = tinycodes_dataset.cast(airoboros_dataset.features)
    # dataset = datasets.concatenate_datasets(
    #   [airoboros_dataset, orca_dataset, platypus_dataset, wizardlm_dataset, tokenbender_dataset])

    # # speechless-reasoning-v0.4-sharegpt.jsonl
    # dataset_file = "/opt/local/datasets/Speechless/speechless-reasoning-v0.4-sharegpt.jsonl"
    # # FIXME airoboros 2.2
    # airoboros_dataset = load_airoboros_22_dataset()
    # orca_dataset = load_orca_dataset("/opt/local/datasets/OpenOrca")
    # platypus_dataset = load_platypus_dataset("/opt/local/datasets/garage-bAInd/Open-Platypus")
    # wizardlm_dataset = load_wizardlm_dataset("/opt/local/datasets/WizardLM/WizardLM_evol_instruct_V2_196k")
    # # FIXME use tokenbender
    # tokenbender_dataset = load_tokenbender_dataset("/opt/local/datasets/TokenBender/python_evol_instruct_51k")
    # # FIXME not use tinycodes
    # # tinycodes_dataset = load_tinycodes_dataset("/opt/local/datasets/code/nampdn-ai/tiny-codes")
    # # tinycodes_dataset = tinycodes_dataset.cast(airoboros_dataset.features)
    # dataset = datasets.concatenate_datasets(
    #   [airoboros_dataset, orca_dataset, platypus_dataset, wizardlm_dataset, tokenbender_dataset])

    # # speechless-agents-v0.1.jsonl
    # dataset_file = "/opt/local/datasets/Speechless/speechless-agents-v0.1.jsonl"
    # # FIXME airoboros 2.2
    # airoboros_dataset = load_airoboros_22_dataset()
    # airoboros_dataset = airoboros_dataset.train_test_split(test_size=10000)['test']

    # orca_dataset = load_orca_dataset("/opt/local/datasets/OpenOrca")
    # orca_dataset = orca_dataset.train_test_split(test_size=10000)['test']

    # platypus_dataset = load_platypus_dataset("/opt/local/datasets/garage-bAInd/Open-Platypus")
    # platypus_dataset = platypus_dataset.train_test_split(test_size=10000)['test']

    # wizardlm_dataset = load_wizardlm_dataset("/opt/local/datasets/WizardLM/WizardLM_evol_instruct_V2_196k")
    # wizardlm_dataset = wizardlm_dataset.train_test_split(test_size=10000)['test']

    # # FIXME use tokenbender
    # tokenbender_dataset = load_tokenbender_dataset("/opt/local/datasets/TokenBender/python_evol_instruct_51k")
    # tokenbender_dataset = tokenbender_dataset.train_test_split(test_size=10000)['test']

    # speechless-agents-v0.2.jsonl
    dataset_file = "/opt/local/datasets/Speechless/speechless-agents-v0.2.jsonl"
    # FIXME airoboros 2.2
    airoboros_dataset = load_airoboros_22_dataset()
    airoboros_dataset = airoboros_dataset.train_test_split(test_size=10000)['test']

    orca_dataset = load_orca_dataset("/opt/local/datasets/OpenOrca")
    orca_dataset = orca_dataset.train_test_split(test_size=10000)['test']

    platypus_dataset = load_platypus_dataset("/opt/local/datasets/garage-bAInd/Open-Platypus")
    platypus_dataset = platypus_dataset.train_test_split(test_size=10000)['test']

    wizardlm_dataset = load_wizardlm_dataset("/opt/local/datasets/WizardLM/WizardLM_evol_instruct_V2_196k")
    wizardlm_dataset = wizardlm_dataset.train_test_split(test_size=10000)['test']

    # FIXME use tokenbender
    tokenbender_dataset = load_tokenbender_dataset("/opt/local/datasets/TokenBender/python_evol_instruct_51k")
    tokenbender_dataset = tokenbender_dataset.train_test_split(test_size=10000)['test']

    # FIXME not use tinycodes
    # tinycodes_dataset = load_tinycodes_dataset("/opt/local/datasets/code/nampdn-ai/tiny-codes")
    # tinycodes_dataset = tinycodes_dataset.cast(airoboros_dataset.features)
    # tinycodes_dataset = tinycodes_dataset.train_test_split(test_size=10000)['test']

    dataset = datasets.concatenate_datasets(
      [airoboros_dataset, orca_dataset, platypus_dataset, wizardlm_dataset, tokenbender_dataset])

    # speechless-reasoning-v1.0.jsonl 136,663 samples
    # dataset_file = "/opt/local/datasets/Speechless/speechless-thoughts-v1.0.jsonl"
    # tokenbender_dataset = load_tokenbender_dataset("/opt/local/datasets/TokenBender/python_evol_instruct_51k")
    # tinycodes_dataset = load_tinycodes_dataset("/opt/local/datasets/code/nampdn-ai/tiny-codes")
    # tinycodes_dataset = tinycodes_dataset.cast(airoboros_dataset.features)
    # # print(f"tinycodes_dataset: {tinycodes_dataset.column_names=}")
    #   [airoboros_dataset, orca_dataset, platypus_dataset, wizardlm_dataset, tokenbender_dataset, tinycodes_dataset])

    print(f"Merging all datasets ...")
    def _get_data_length(item):
        prompt = f"{tokenizer.bos_token}{item['instruction']}{item['response']}{tokenizer.eos_token}"
        return len(
            tokenizer(
                prompt,
                # max_length=model_max_len + 1,
                truncation=False,
                add_special_tokens=False
            ).input_ids
        )

    # def _filter_by_token_length(item):
    #     tok_len =  _get_data_length(item)
    #     return tok_len >= 128 and tok_len <= model_max_len - 50

    # total_samples = len(dataset)
    # dataset = dataset.filter(_filter_by_token_length)
    # remaining_samples = len(dataset)
    # print(f"Filter dataset from {total_samples} to {remaining_samples}. {remaining_samples/total_samples * 100:.2f}%")

    print(f"Convert to sharegpt format ...")
    dataset = dataset.map(lambda x: {
        # 'system_prompt': x['system'],
        'system_prompt': "", #x['system'],
        'category': x['category'],
        'dialog': [
                {
                    'from': 'human',
                    'value': x['instruction'],  
                },
                {
                    'from': 'gpt',
                    'value': x['response'],  
                },
        ]
    })
    # Remove unused columns.
    dataset = dataset.remove_columns(
        # FIXME
        [col for col in dataset.column_names if col not in ['dialog', 'system_prompt', 'category']]
    )
    # print(dataset[0])

    agent_instruct_dataset = load_agent_instruct_dataset()
    # print(agent_instruct_dataset[0])

    dataset = datasets.concatenate_datasets([dataset, agent_instruct_dataset])

    dataset.to_json(dataset_file, orient="records", lines=True, index=False)
    print(f"Saved {len(dataset)} samples to {dataset_file}")

def main():
    model_name_or_path="/opt/local/llm_models/huggingface.co/llm_agents/tora-code-7b-v1.0"
    model_max_len = 1024 * 32
    prepare_data(model_name_or_path=model_name_or_path, model_max_len=model_max_len)
  
if __name__ == '__main__':
    main()