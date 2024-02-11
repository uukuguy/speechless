import os, re, copy
import pandas as pd
from typing import Dict, Sequence
from loguru import logger
import torch
from torch.nn.utils.rnn import pad_sequence
import transformers
from datasets import load_dataset, Dataset
from dataclasses import dataclass

IGNORE_INDEX = -100


@dataclass
class DataCollatorForCausalLM(object):
    tokenizer: transformers.PreTrainedTokenizer
    model_max_length: int

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract elements
        sources = [f"{self.tokenizer.bos_token}{example['input']}" for example in instances]
        targets = [f"{example['output']}{self.tokenizer.eos_token}" for example in instances]

        # Tokenize
        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=self.model_max_length,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_targets = self.tokenizer(
            targets,
            max_length=self.model_max_length,
            truncation=True,
            add_special_tokens=False,
        )

        # Build the input and labels for causal LM
        input_ids = []
        labels = []
        for tokenized_source, tokenized_target in zip(
            tokenized_sources_with_prompt['input_ids'], tokenized_targets['input_ids']
        ):
            input_ids.append(torch.tensor(tokenized_source + tokenized_target))
            labels.append(
                torch.tensor([IGNORE_INDEX for _ in range(len(tokenized_source))] + copy.deepcopy(tokenized_target))
            )
        # Apply padding
        if self.tokenizer.padding_side == "left":
            input_ids = [t.flip(-1) for t in input_ids]
            labels = [t.flip(-1) for t in labels]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        if self.tokenizer.padding_side == "left":
            input_ids = input_ids.flip(-1)
            labels = labels.flip(-1)

        data_dict = {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': input_ids.ne(self.tokenizer.pad_token_id),
        }
        return data_dict


ALPACA_PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response: "
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: "
    ),
}

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


def preprocess_toolbench_dataset(
    example,
    model_max_length: int,
    tokenizer: transformers.PreTrainedTokenizer,
    template: str = "tool-llama-single-round"
) -> Dict:
    # conv = get_conversation_template(template)
    # if template == "tool-llama":
    #     roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    # elif template == "tool-llama-single-round" or template == "tool-llama-multi-rounds":
    #     roles = {"system": conv.roles[0], "user": conv.roles[1], "function": conv.roles[2], "assistant": conv.roles[3]}

    roles = {
        "system": "System",
        "user": "User",
        "function": "Function",
        "assistant": "Assistant"
    }
    seps = ["\n", "</s>"]

    # Apply prompt templates
    conversation = ""
    for i, round in enumerate(example['conversations']):
        role = roles[round['from']]
        message = round['value']
        if i + 1 == len(example['conversations']) and message:
            conversation += role + ": " + str(message) + seps[1]
        elif message:
            conversation += role + ": " + str(message) + seps[0]
        else:
            conversation += role + ":"

    # Tokenize conversations
    input_ids = tokenizer(
        conversation,
        # return_tensors="pt",
        # padding="max_length",
        max_length=model_max_length,
        truncation=True,
        # add_special_tokens=False,
    ).input_ids
    # input_ids begin with '<s>' and end with '</s>'
    assert input_ids[0] == tokenizer.bos_token_id and input_ids[-1] == tokenizer.eos_token_id
    labels = copy.deepcopy(input_ids)

    input_ids = torch.tensor(input_ids)
    labels = torch.tensor(labels)

    # Mask targets. Only compute loss on the assistant outputs.
    sep = seps[0] + roles['assistant'] + ": "

    total_len = int(labels.ne(tokenizer.pad_token_id).sum())
    turns = conversation.split(seps[1])
    cur_len = 1
    labels[:cur_len] = IGNORE_INDEX
    for i, turn in enumerate(turns):
        if turn == "":
            continue
        turn_len = len(tokenizer(turn).input_ids)

        parts = turn.split(sep)

        # only train on the last assistant reply, treat the history chat as instruction
        prefix = parts[:-1]
        instruction = ""
        for part in prefix:
            instruction += part
            instruction += sep

        # "-2" is hardcoded for the LLaMA tokenizer to make the offset correct.
        instruction_len = len(tokenizer(instruction).input_ids) - 2

        # Ignore the user instructions
        labels[cur_len:cur_len + instruction_len] = IGNORE_INDEX
        cur_len += turn_len

    labels[cur_len:] = IGNORE_INDEX

    # if False:  # Inspect and check the correctness of masking
    #     z = labels.clone()
    #     z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
    #     rank0_print(tokenizer.decode(z))

    if cur_len < model_max_length:
        if cur_len != total_len:
            labels[:] = IGNORE_INDEX
            logger.warning(
                f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                f" (ignored)"
                f"{conversation=}"
            )
    return dict(
        input_ids=input_ids,
        labels=labels,
        # attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


def preprocess_multi_rounds_dialog(
    example,
    model_max_length: int,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # conv = get_conversation_template(template)
    # if template == "tool-llama":
    #     roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    # elif template == "tool-llama-single-round" or template == "tool-llama-multi-rounds":
    #     roles = {"system": conv.roles[0], "user": conv.roles[1], "function": conv.roles[2], "assistant": conv.roles[3]}

    roles = {
        "system": "System",
        "user": "User",
        "function": "Function",
        "assistant": "Assistant"
    }
    seps = ["\n", "</s>"]

    # The dialogue process is divided into multiple rounds, with each round ending when the Assistant speaks.
    dialog = example['conversations']
    dialog_rounds = []
    round_messages = []
    for i, round in enumerate(dialog):
        who = round['from']
        message = round['value']
        if who != 'assistant':
            round_messages.append((who, message))
        else:
            dialog_rounds.append({
                'round_messages': round_messages,
                'assistant': message,
            })
            round_messages = []
    if len(round_messages) > 0:
        logger.warning(f"WARNING: the last round is not ended by the assistant. IGNORE!!!. {dialog=}")
        dialog_rounds = []
    # print(f"{dialog_rounds=}")

    example_input_ids = None
    example_output_ids = None
    for idx, round in enumerate(dialog_rounds):
        round_messages = round['round_messages']
        assistant_message = round['assistant']
        source = ""
        for (who, message) in round_messages:
            source += roles[who] + ": " + str(message) + seps[0]
        source += roles['assistant'] + ": "
        target = assistant_message + tokenizer.eos_token

        # source = f"{tokenizer.bos_token}{source}"
        # target = f"{bot_response.strip()}\n{tokenizer.eos_token}"

        tokenized_source = tokenizer(source, max_length=model_max_length, truncation=True, add_special_tokens=False)
        tokenized_target = tokenizer(target, max_length=model_max_length, truncation=True, add_special_tokens=False)
        tokenized_input = torch.tensor(tokenized_source['input_ids'] + tokenized_target['input_ids'])
        tokenized_output = torch.tensor([IGNORE_INDEX for _ in range(len(tokenized_source['input_ids']))] +
                                        copy.deepcopy(tokenized_target['input_ids']))
        if idx == 0:
            example_input_ids = tokenized_input
            example_output_ids = tokenized_output
        else:
            example_input_ids = torch.concatenate((example_input_ids, tokenized_input), dim=0)
            example_output_ids = torch.concatenate((example_output_ids, tokenized_output), dim=0)

    input_ids = example_input_ids
    labels = example_output_ids
    return dict(
        input_ids=input_ids,
        labels=labels,
        # attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


def generate_round_prompt_toolllama(
    idx: int,
    human_input: str,
    bot_response: str,
    bos_token: str,
    eos_token: str,
    system_prompt: str = None,
):
    if idx == 0:
        if system_prompt:
            source = f"{system_prompt} Human: {human_input} Assistant: "
        else:
            system_prompt = "A chat between a curious user and an artificial intelligence assistant who can use external tools and APIs to solve the user's question."
            "The assistant gives tools and APIs calling processes or final answer to the human's question."
            human_input = "Human: {instruction} Assistant: ".format(instruction=human_input)
            source = f"{system_prompt} {human_input}"
    else:
        human_input = "Human: {instruction} Assistant: ".format(instruction=human_input)
        source = f"{human_input}"
    source = f"{bos_token}{source}"
    target = f"{bot_response.strip()}\n{eos_token}"

    return source, target


def generate_round_prompt_alpaca(
    idx: int,
    human_input: str,
    bot_response: str,
    bos_token: str,
    eos_token: str,
    system_prompt: str = None,
):
    if idx == 0:
        if system_prompt:
            source = f"{system_prompt}\n\n### Instruction:\n{human_input}\n\n### Response:"
        else:
            system_prompt = "Below is an instruction that describes a task.\nWrite a response that appropriately completes the request.\n\n"
            human_input = "### Instruction:\n{instruction}\n\n### Response:".format(instruction=human_input)
            source = f"{system_prompt}{human_input}"
    else:
        human_input = "### Instruction:\n{instruction}\n\n### Response:".format(instruction=human_input)
        source = f"{human_input}"

    source = f"{bos_token}{source}"
    target = f"{bot_response.strip()}\n{eos_token}"

    return source, target


def generate_round_prompt_llama2(
    idx: int,
    human_input: str,
    bot_response: str,
    bos_token: str,
    eos_token: str,
    system_prompt: str = None,
):
    if idx == 0:
        if system_prompt:
            source = f"{bos_token}[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n{human_input}[/INST]"
        else:
            source = f"{bos_token}[INST]{human_input}[/INST]"
    else:
        source = f"[INST]{human_input}[/INST]"

    target = f"{bot_response.strip()}\n{eos_token}"

    return source, target


def generate_round_prompt_chatlm(
    idx: int,
    human_input: str,
    bot_response: str,
    bos_token: str,
    eos_token: str,
    system_prompt: str = None,
):
    # f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant"
    if idx == 0:
        # if system_prompt:
        #     # source = f"{system_prompt}\n\n### Instruction:\n{human_input}\n\n### Response:"
        #     source = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        # else:
        #     # system_prompt = "Below is an instruction that describes a task.\nWrite a response that appropriately completes the request.\n\n"
        system_prompt = "You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior."
        human_input = f"<|im_start|>user\n{human_input}<|im_end|>\n<|im_start|>assistant"
        source = f"<|im_start|>system\n{system_prompt}<|im_end|>\n{human_input}"
    else:
        human_input = f"<|im_start|>user\n{human_input}<|im_end|>\n<|im_start|>assistant"
        source = f"{human_input}"

    source = f"{bos_token}{source}"
    target = f"{bot_response.strip()}\n{eos_token}"

    return source, target

def generate_round_prompt_minicpm(
    idx: int,
    human_input: str,
    bot_response: str,
    bos_token: str,
    eos_token: str,
    system_prompt: str = None,
):
    source = f"{bos_token}<用户>{human_input}<AI>"
    target = f"{bot_response.strip()}{eos_token}"

    return source, target


@dataclass
class DialogDataCollatorForCausalLM(object):
    tokenizer: transformers.PreTrainedTokenizer
    model_max_length: int
    prompt_type: str = None

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract elements
        input_ids = []
        labels = []
        for example in instances:
            # system_prompt = example.get('system_prompt', 'A Bot').strip() + "\n\n"
            if self.prompt_type is not None:
                prompt_type = self.prompt_type
            else:
                prompt_type = example.get('prompt_type', None)
            if prompt_type == "tool-llama-single-round":
                data_dict = preprocess_toolbench_dataset(
                    example,
                    model_max_length=self.model_max_length,
                    tokenizer=self.tokenizer,
                    template="tool-llama-single-round"
                )
                example_input_ids = data_dict['input_ids']
                example_labels = data_dict['labels']
                if example_input_ids is not None:
                    # print(f"{example_input_ids.shape=},{example_input_ids=}")
                    # print(f"{example_labels.shape=},{example_labels=}")
                    input_ids.append(example_input_ids)
                    labels.append(example_labels)
                continue

            elif prompt_type == "tool-llama-multi-rounds":
                data_dict = preprocess_multi_rounds_dialog(
                    example,
                    model_max_length=self.model_max_length,
                    tokenizer=self.tokenizer,
                )
                example_input_ids = data_dict['input_ids']
                example_labels = data_dict['labels']
                if example_input_ids is not None:
                    # print(f"{example_input_ids.shape=},{example_input_ids=}")
                    # print(f"{example_labels.shape=},{example_labels=}")
                    input_ids.append(example_input_ids)
                    labels.append(example_labels)
                continue

            system_prompt = example.get('system_prompt', "").strip()
            if system_prompt:
                system_prompt += "\n\n"
            example_input_ids = None
            example_output_ids = None

            # human_bot_dialog = example['dialog']
            human_bot_dialog = []
            dialog = example['conversations']
            for _i in range(len(dialog) // 2):
                human_input = dialog[2 * _i]['value']
                bot_output = dialog[2 * _i + 1]['value']
                human_bot_dialog.append((human_input, bot_output))
            if len(human_bot_dialog) < 1:
                continue
            for idx, round in enumerate(human_bot_dialog):
                human_input, bot_response = round
                if prompt_type == 'toolllama':
                    source, target = generate_round_prompt_toolllama(
                        idx,
                        human_input,
                        bot_response,
                        bos_token=self.tokenizer.bos_token,
                        eos_token=self.tokenizer.eos_token,
                        system_prompt=system_prompt,
                    )
                elif prompt_type == "chatlm":
                    source, target = generate_round_prompt_chatlm(
                        idx,
                        human_input,
                        bot_response,
                        bos_token=self.tokenizer.bos_token,
                        eos_token=self.tokenizer.eos_token,
                        system_prompt=system_prompt,
                    )
                elif prompt_type == "llama2":
                    source, target = generate_round_prompt_llama2(
                        idx,
                        human_input,
                        bot_response,
                        bos_token=self.tokenizer.bos_token,
                        eos_token=self.tokenizer.eos_token,
                        system_prompt=system_prompt,
                    )
                elif prompt_type == "minicpm":
                    source, target = generate_round_prompt_minicpm(
                        idx,
                        human_input,
                        bot_response,
                        bos_token=self.tokenizer.bos_token,
                        eos_token=self.tokenizer.eos_token,
                        system_prompt=system_prompt,
                    )
                else:  # default alpaca
                    source, target = generate_round_prompt_alpaca(
                        idx,
                        human_input,
                        bot_response,
                        bos_token=self.tokenizer.bos_token,
                        eos_token=self.tokenizer.eos_token,
                        system_prompt=system_prompt,
                    )

                tokenized_source = self.tokenizer(
                    source,
                    max_length=self.model_max_length,
                    truncation=True,
                    add_special_tokens=False,
                )
                tokenized_target = self.tokenizer(
                    target,
                    max_length=self.model_max_length,
                    truncation=True,
                    add_special_tokens=False,
                )
                tokenized_input = torch.tensor(tokenized_source['input_ids'] + tokenized_target['input_ids'])
                tokenized_output = torch.tensor([IGNORE_INDEX for _ in range(len(tokenized_source['input_ids']))] +
                                                copy.deepcopy(tokenized_target['input_ids']))

                # print(f"{source=}")
                # print(f"{tokenized_input=}")
                # print(f"{target=}")
                # print(f"{tokenized_target=}")
                if idx == 0:
                    example_input_ids = tokenized_input
                    example_output_ids = tokenized_output
                else:
                    example_input_ids = torch.concatenate((example_input_ids, tokenized_input), dim=0)
                    example_output_ids = torch.concatenate((example_output_ids, tokenized_output), dim=0)

            input_ids.append(example_input_ids)
            labels.append(example_output_ids)

        # print(f"{example=}")
        # print(f"{input_ids=}")
        # print(f"{labels=}")
        # Apply padding
        if self.tokenizer.padding_side == "left":
            input_ids = [t.flip(-1) for t in input_ids]
            labels = [t.flip(-1) for t in labels]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        if self.tokenizer.padding_side == "left":
            input_ids = input_ids.flip(-1)
            labels = labels.flip(-1)

        data_dict = {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': input_ids.ne(self.tokenizer.pad_token_id),
        }
        return data_dict


def extract_unnatural_instructions_data(examples, extract_reformulations=False):
    out = {
        'input': [],
        'output': [],
    }
    for example_instances in examples['instances']:
        for instance in example_instances:
            out['input'].append(instance['instruction_with_input'])
            out['output'].append(instance['output'])
    if extract_reformulations:
        for example_reformulations in examples['reformulations']:
            if example_reformulations is not None:
                for instance in example_reformulations:
                    out['input'].append(instance['instruction_with_input'])
                    out['output'].append(instance['output'])
    return out


def extract_alpaca_dataset(example):
    if example.get("input", "") != "":
        prompt_format = ALPACA_PROMPT_DICT["prompt_input"]
    else:
        prompt_format = ALPACA_PROMPT_DICT["prompt_no_input"]
    return {
        'input': prompt_format.format(**example)
    }


def local_dataset(dataset_name, test_size=0.02):
    if dataset_name.endswith(('.json', '.jsonl')):
        full_dataset = Dataset.from_json(path_or_paths=dataset_name)
    elif dataset_name.endswith('.csv'):
        full_dataset = Dataset.from_pandas(pd.read_csv(dataset_name))
    elif dataset_name.endswith('.tsv'):
        full_dataset = Dataset.from_pandas(pd.read_csv(dataset_name, delimiter='\t'))
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_name}")

    if 'category' in full_dataset.column_names:
        full_dataset = full_dataset.class_encode_column('category')
        return full_dataset.train_test_split(test_size=test_size, stratify_by_column='category')
    return full_dataset.train_test_split(test_size=test_size)


class RepeatDataset():

    def __init__(self, ds, repeat_batch_size, repeat_steps):
        self.ds = ds
        self.batch_size = repeat_batch_size * repeat_steps
        self.in_cache = []
        self.out_cache = []
        self.first_count = 0

    def __len__(self):
        return len(self.ds) * 2

    def __getitem__(self, idx):
        # new_idx = self.get_new_idx(idx)
        new_idx = idx % self.batch_size
        self.in_cache.append(new_idx)

        if self.first_count < self.batch_size:
            self.first_count += 1
            ret_idx = self.in_cache.pop(0)
            self.out_cache.append(ret_idx)
        elif self.first_count < self.batch_size * 2:
            self.first_count += 1
            ret_idx = self.out_cache.pop(0)
        else:
            self.first_count = 0
            ret_idx = self.in_cache.pop(0)
            self.out_cache.append(ret_idx)

        return self.ds[ret_idx]

    def get_new_idx(self, idx):
        n = idx // (self.batch_size * 2)
        d = idx % (self.batch_size * 2)
        if n < len(self.ds) // self.batch_size:
            new_idx = self.batch_size * n + d % self.batch_size
        else:
            d0 = len(self.ds) % self.batch_size
            if d0 > 0:
                new_idx = self.batch_size * n + d % d0
            else:
                new_idx = self.batch_size * n + d % self.batch_size
        assert new_idx < len(self.ds), f"{idx=}, {new_idx=}, {len(self.ds)=}, {self.batch_size=}, {n=}, {d=}"
        return new_idx

    # def __getitem__(self, idx):
    #     new_idx = self.get_new_idx(idx)
    #     return self.ds[new_idx]


def make_data_module(tokenizer: transformers.PreTrainedTokenizer, args) -> Dict:
    """
    Make dataset and collator for supervised fine-tuning.
    Datasets are expected to have the following columns: { `input`, `output` }

    Available datasets to be selected with `dataset` argument:
        - alpaca, 52002 examples
        - alpaca cleaned, 51942 examples
        - chip2 (OIG), 210289 examples
        - self-instruct, 82612 examples
        - hh-rlhf (Anthropic), 160800 examples
        - longform, 23.7k examples
        - oasst1 (OpenAssistant) primary message tree only, 9,846 examples

    Coming soon:
        - unnatural instructions core, 66010 examples
        - unnatural instructions full, 240670 examples
        - alpaca-gpt4, 52002 examples
        - unnatural-instructions-gpt4, 9000 examples
        - supernatural-instructions, 69624 examples (same as paper with 100 ex/task more can be used)
        - flan (FLAN v2), up to 20M examples available
        - vicuna

    """

    def load_data(dataset_name):
        if dataset_name == 'alpaca':
            return load_dataset("tatsu-lab/alpaca")
        elif dataset_name == 'alpaca-clean':
            return load_dataset("yahma/alpaca-cleaned")
        elif dataset_name == 'chip2':
            return load_dataset("laion/OIG", data_files='unified_chip2.jsonl')
        elif dataset_name == 'self-instruct':
            return load_dataset("yizhongw/self_instruct", name='self_instruct')
        elif dataset_name == 'hh-rlhf':
            return load_dataset("Anthropic/hh-rlhf")
        elif dataset_name == 'longform':
            return load_dataset("akoksal/LongForm")
        elif dataset_name == 'oasst1':
            return load_dataset("timdettmers/openassistant-guanaco")
        elif dataset_name == 'vicuna':
            raise NotImplementedError("Vicuna data was not released.")
        else:
            if os.path.exists(dataset_name):
                try:
                    args.dataset_format = args.dataset_format if args.dataset_format else "input-output"
                    full_dataset = local_dataset(dataset_name, test_size=args.eval_dataset_size)
                    return full_dataset
                except:
                    raise ValueError(f"Error loading dataset from {dataset_name}")
            else:
                raise NotImplementedError(f"Dataset {dataset_name} not implemented yet.")

    def format_dataset(dataset, dataset_format):
        if (
            dataset_format == 'alpaca' or dataset_format == 'alpaca-clean'
            or (dataset_format is None and args.dataset in ['alpaca', 'alpaca-clean'])
        ):
            dataset = dataset.map(extract_alpaca_dataset, remove_columns=['instruction'])
            dataset = dataset.map(lambda x: {'conversations': [(x['input'], x['output'])]})

        elif dataset_format == 'chip2' or (dataset_format is None and args.dataset == 'chip2'):
            # dataset = dataset.map(lambda x: {
            #     'input': x['text'].split('\n<bot>: ')[0].replace('<human>: ', ''),
            #     'output': x['text'].split('\n<bot>: ')[1],
            # })
            dataset = dataset.map(
                lambda x: {
                    'conversations':
                    [(x['text'].split('\n<bot>: ')[0].replace('<human>: ', ''), x['text'].split('\n<bot>: ')[1])]
                }
            )
        elif dataset_format == 'self-instruct' or (dataset_format is None and args.dataset == 'self-instruct'):
            # for old, new in [["prompt", "input"], ["completion", "output"]]:
            #     dataset = dataset.rename_column(old, new)
            dataset = dataset.map(lambda x: {'conversations': [(x['prompt'], x['completion'])]})
        elif dataset_format == 'hh-rlhf' or (dataset_format is None and args.dataset == 'hh-rlhf'):
            # dataset = dataset.map(lambda x: {
            #     'input': '',
            #     'output': x['chosen']
            # })
            dataset = dataset.map(lambda x: {'conversations': [('', x['chosen'])]})
        elif dataset_format == 'oasst1' or (dataset_format is None and args.dataset == 'oasst1'):
            # dataset = dataset.map(lambda x: {
            #     'input': '',
            #     'output': x['text'],
            # })
            dataset = dataset.map(lambda x: {'conversations': [('', x['text'])]})
        elif dataset_format == 'airoboros':
            logger.info("---------- Formatting dataset for Airoboros. ----------")

            def _format_airoboros(instruction):
                # FIXME - for Spider prompt
                if "### Instructions:" in instruction["instruction"]:
                    in_ = instruction["instruction"]
                    out_ = instruction['response']
                    # return {
                    #     'input': in_,
                    #     'output': out_,
                    # }
                    return {
                        'conversations': [(in_, out_)]
                    }
                else:
                    in_ = None
                    if instruction.get("skip_prompt_formatting"):
                        in_ = instruction["instruction"].strip() + "\n"
                    else:
                        in_ = "\n".join([
                            (instruction.get('system') or 'A chat.').strip(),
                            f"USER: {instruction['instruction'].strip()}",
                        ])
                        if in_.endswith("PLAINFORMAT"):
                            in_ = re.sub(r"\s+PLAINFORMAT$", "", in_, re.DOTALL)
                            in_ += " PLAINFORMAT"
                        in_ = "\n".join([in_.strip(), "ASSISTANT: "])
                    # return {
                    #     'input': in_,
                    #     'output': instruction['response'].strip() + "\n",
                    # }
                    return {
                        'conversations': [(in_, instruction['response'].strip() + "\n")]
                    }

            dataset = dataset.map(_format_airoboros)
        elif dataset_format == 'mistral':
            logger.info("---------- Formatting dataset for Mistral. ----------")

            def _format_mistral(instruction):
                # FIXME - for Spider prompt
                if "### Instructions:" in instruction["instruction"]:
                    in_ = instruction["instruction"]
                    inst = instruction["instruction"]
                    toks = inst.split("### Input:\n")
                    if len(toks) == 2:
                        first = toks[0]
                        first = first.replace("### Instructions:\n", "")
                        second = toks[1]
                        second_toks = second.split("### Response:\n")
                        if len(second_toks) == 2:
                            input = second_toks[0]
                            response = second_toks[1]
                            in_ = "<s>[INST] " + first + " [/INST]\n" + input + "</s> " + "[INST] " + response + " [/INST]"

                    out_ = instruction['response'] + "</s>"
                    # print(f"{in_=}")
                    # print(f"{out_=}")
                    # return {
                    #     'input': in_,
                    #     'output': out_,
                    # }
                    return {
                        'conversations': [(in_, out_)]
                    }
                else:
                    in_ = f"<s>[INST] {instruction['instruction']} [/INST]"
                    out_ = f"{instruction['response']}</s>"
                    # return {
                    #     'input': in_,
                    #     'output': out_,
                    # }
                    return {
                        'conversations': [(in_, out_)]
                    }

            dataset = dataset.map(_format_mistral)
        elif dataset_format == 'llama2':
            logger.info("---------- Formatting dataset for Llama2. ----------")

            def _format_llama2(instruction):
                sys_msg = instruction.get('system', 'A chat.')
                user_msg = instruction['instruction']
                mode_msg = instruction['response']
                in_ = f"<s>[INST] <<SYS>>\n{sys_msg}\n<</SYS>>\n{user_msg}[/INST]"
                out_ = f"{instruction['response']}</s>"
                # return {
                #     'input': in_,
                #     'output': out_,
                # }
                return {
                    'conversations': [(in_, out_)]
                }

            dataset = dataset.map(_format_llama2)

        elif dataset_format == 'instruction-input-response':

            def _format_instruction_input_response(example):
                if example.get("input", "") != "":
                    in_ = PROMPT_DICT["prompt_input"].format(instruction=example["instruction"], input=example["input"])
                else:
                    in_ = PROMPT_DICT["prompt_no_input"].format(instruction=example["instruction"])
                out_ = f"{example['response']}"
                # out_lines = out_.strip().split("\n")
                # if len(out_lines) > 1:
                #     if out_lines[0].startswith("```"):
                #         in_ += out_lines[0] + "\n"
                #         out_ = "\n".join(out_lines[1:])

                # return {'input': in_,
                #         'output': out_}
                return {
                    'conversations': [(in_, out_)]
                }

            dataset = dataset.map(_format_instruction_input_response)

        elif dataset_format == 'input-output':
            # leave as is
            pass

            def _format_input_output(instruction):
                # return {
                #     'input': instruction['instruction'],
                #     'output': instruction['response'],
                # }
                return {
                    'conversations': [(instruction['instruction'], instruction['response'])]
                }

            dataset = dataset.map(_format_input_output)
        elif dataset_format == 'conversations':

            def _format_multi_turns(example):
                human_bot_dialog = []
                dialog = example['conversations']
                for round in dialog:
                    who = round['from']
                    response = round['value']
                    human_bot_dialog.append({
                        "from": who,
                        "value": response,
                    })
                return {
                    'conversations': human_bot_dialog
                }

            dataset = dataset.map(_format_multi_turns)

        # Remove unused columns.
        dataset = dataset.remove_columns(
            # FIXME
            # [col for col in dataset.column_names['train'] if col not in ['input', 'output']]
            [
                col for col in dataset.column_names['train']
                if col not in ['conversations', 'system_prompt', 'prompt_type']
            ]
        )
        return dataset

    # Load dataset.
    dataset = load_data(args.dataset)
    dataset = format_dataset(dataset, args.dataset_format)

    # Split train/eval, reduce size
    logger.info(f"---------- Splitting dataset into train/eval ----------")
    if args.do_eval or args.do_predict:
        if 'eval' in dataset:
            eval_dataset = dataset['eval']
        elif 'test' in dataset:
            eval_dataset = dataset['test']
        else:
            logger.info('Splitting train dataset in train and validation according to `eval_dataset_size`')
            if 'category' in dataset["train"].column_names:
                dataset["train"] = dataset["train"].class_encode_column('category')
                dataset = dataset["train"].train_test_split(
                    test_size=args.eval_dataset_size, stratify_by_column='category', seed=args.seed
                )
            else:
                dataset = dataset["train"].train_test_split(
                    test_size=args.eval_dataset_size, shuffle=True, seed=args.seed
                )
            eval_dataset = dataset['test']
        if args.max_eval_samples is not None and len(eval_dataset) > args.max_eval_samples:
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))
        if args.group_by_length:
            eval_dataset = eval_dataset.map(lambda x: {'length': len(x['input']) + len(x['output'])})
    if args.do_train:
        train_dataset = dataset['train']
        max_train_samples = args.max_train_samples
        if max_train_samples is not None and max_train_samples > 0 and max_train_samples < 1.0:
            max_train_samples = int(len(train_dataset) * max_train_samples)
        else:
            max_train_samples = 0.0
        if max_train_samples >= 1.0 and len(train_dataset) > max_train_samples:
            train_dataset = train_dataset.select(range(max_train_samples))

        if args.group_by_length:
            train_dataset = train_dataset.map(lambda x: {'length': len(x['input']) + len(x['output'])})

    # # Remove any training data that exceeds the max length.
    # def _get_data_length(item):
    #     prompt = f"{tokenizer.bos_token}{item['input']}{item['output']}{tokenizer.eos_token}"
    #     return len(
    #         tokenizer(prompt, max_length=args.model_max_length + 1, truncation=True, add_special_tokens=False).input_ids
    #     )

    # if args.force_remove_overlength_samples:
    #     logger.info(f"---------- Filtering out samples longer than {args.model_max_length} ----------")
    #     prev_len = len(train_dataset)
    #     train_dataset = train_dataset.filter(lambda x: _get_data_length(x) < args.model_max_length - 10)
    #     logger.info(f"Filtered out {prev_len - len(train_dataset)} samples. ({len(train_dataset)}/{prev_len})")

    # FIXME
    # data_collator = DataCollatorForCausalLM(
    data_collator = DialogDataCollatorForCausalLM(
        tokenizer=tokenizer,
        model_max_length=args.model_max_length,
        prompt_type=args.prompt_type,
    )

    # # FIXME
    # if args.repeat_steps > 0:
    #     one_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps * torch.cuda.device_count()
    #     train_dataset = RepeatDataset(train_dataset, repeat_batch_size=one_batch_size, repeat_steps=args.repeat_steps)

    return dict(
        train_dataset=train_dataset if args.do_train else None,
        eval_dataset=eval_dataset if args.do_eval else None,
        predict_dataset=eval_dataset if args.do_predict else None,
        data_collator=data_collator
    )
