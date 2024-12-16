import os, re
import pandas as pd
from typing import Dict, List, Union
from loguru import logger
import transformers
from datasets import load_dataset, Dataset
from .data_collectors import DialogDataCollatorForCausalLM

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


def extract_alpaca_dataset(example):
    if example.get("input", "") != "":
        prompt_format = ALPACA_PROMPT_DICT["prompt_input"]
    else:
        prompt_format = ALPACA_PROMPT_DICT["prompt_no_input"]
    return {
        'input': prompt_format.format(**example)
    }


from datasets import concatenate_datasets, interleave_datasets, Dataset, IterableDataset
from ..finetune_qlora_arguments import DataArguments, TrainingArguments
def merge_dataset(
    all_datasets: List[Union["Dataset", "IterableDataset"]],
    args,
    # data_args: "DataArguments",
    # training_args: "TrainingArguments",
) -> Union["Dataset", "IterableDataset"]:
    if len(all_datasets) == 1:
        return all_datasets[0]
    elif args.mix_strategy == "concat":
        if args.streaming:
            logger.warning("The samples between different datasets will not be mixed in streaming mode.")
        return concatenate_datasets(all_datasets)
    elif args.mix_strategy.startswith("interleave"):
        if not args.streaming:
            logger.warning("We recommend using `mix_strategy=concat` in non-streaming mode.")
        return interleave_datasets(
            datasets=all_datasets,
            probabilities=args.interleave_probs,
            seed=args.seed,
            stopping_strategy="first_exhausted" if args.mix_strategy.endswith("under") else "all_exhausted",
        )
    else:
        raise ValueError("Unknown mixing strategy.")

def local_dataset(dataset_name, args, test_size=0.02):
    if "," in dataset_name:
        all_datasets = []
        for dataset_attr in [ n.strip() for n in dataset_name.split(',')]:
            all_datasets.append(load_dataset("json", data_files=dataset_attr))
        # dataset = merge_dataset(all_datasets, data_args, training_args)
        full_dataset = merge_dataset(all_datasets, args)
    else:
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
        return full_dataset.train_test_split(test_size=int(test_size), stratify_by_column='category')
    return full_dataset.train_test_split(test_size=int(test_size))


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
                    full_dataset = local_dataset(dataset_name, args, test_size=args.eval_dataset_size)
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
            dataset = dataset.map(
                lambda x: {
                    'conversations':
                    [(x['text'].split('\n<bot>: ')[0].replace('<human>: ', ''), x['text'].split('\n<bot>: ')[1])]
                }
            )
        elif dataset_format == 'self-instruct' or (dataset_format is None and args.dataset == 'self-instruct'):
            dataset = dataset.map(lambda x: {'conversations': [(x['prompt'], x['completion'])]})
        elif dataset_format == 'hh-rlhf' or (dataset_format is None and args.dataset == 'hh-rlhf'):
            dataset = dataset.map(lambda x: {'conversations': [('', x['chosen'])]})
        elif dataset_format == 'oasst1' or (dataset_format is None and args.dataset == 'oasst1'):
            dataset = dataset.map(lambda x: {'conversations': [('', x['text'])]})
        elif dataset_format == 'airoboros':
            logger.info("---------- Formatting dataset for Airoboros. ----------")

            def _format_airoboros(instruction):
                # FIXME - for Spider prompt
                if "### Instructions:" in instruction["instruction"]:
                    in_ = instruction["instruction"]
                    out_ = instruction['response']
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
                    return {
                        'conversations': [(in_, out_)]
                    }
                else:
                    in_ = f"<s>[INST] {instruction['instruction']} [/INST]"
                    out_ = f"{instruction['response']}</s>"
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
                return {
                    'conversations': [(in_, out_)]
                }

            dataset = dataset.map(_format_instruction_input_response)

        elif dataset_format == 'input-output':
            # leave as is
            pass

            def _format_input_output(instruction):
                # return {
                #     'conversations': [(instruction['instruction'], instruction['response'])]
                # }
                return {
                    'conversations': [
                        {
                            "from": "human", 
                            "value": instruction['instruction'] 
                        },
                        {
                            "from": "assistant", 
                            "value": instruction['response'] 
                        },
                        ]
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

    print(f"{len(train_dataset)} training samples, {len(eval_dataset)} evaluation samples.")

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

    return dict(
        train_dataset=train_dataset if args.do_train else None,
        eval_dataset=eval_dataset if args.do_eval else None,
        predict_dataset=eval_dataset if args.do_predict else None,
        data_collator=data_collator
    )
