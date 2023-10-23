#!/usr/bin/env python
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import math
import gc, ctypes
#from rich import print

# if os.environ.get('ENABLE_FLASH_ATTENTION', 'False') == 'True': 
#     from flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
#     replace_llama_attn_with_flash_attn(packed=True)
#     print(f"Enabled flash attention monkey patching.")

# if os.environ.get('ENABLE_REROPE', "False") == 'True':
#     from rerope_monkey_patch import replace_llama_attention_forword_with_rerope
#     replace_llama_attention_forword_with_rerope(training_length=4096, window=1536)
#     print(f"Enabled rerope monkey patching.")

import random
from collections import defaultdict
import copy
import json
import os
import re
from os.path import exists, join, isdir
from dataclasses import dataclass, field
import sys
from typing import Optional, Dict, Sequence
import numpy as np
from tqdm import tqdm
import logging
import bitsandbytes as bnb
import pandas as pd

# logger = logging.getLogger(__name__)
from loguru import logger


import torch
import transformers
from torch.nn.utils.rnn import pad_sequence
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
    Seq2SeqTrainer,
    BitsAndBytesConfig,
    LlamaTokenizer

)
from datasets import load_dataset, Dataset
import evaluate

from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel
)
from peft.tuners.lora import LoraLayer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR


torch.backends.cuda.matmul.allow_tf32 = True

IGNORE_INDEX = -100

def clean_memory():
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)
    torch.cuda.empty_cache()

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="EleutherAI/pythia-12b"
    )
    trust_remote_code: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."}
    )
    # use_auth_token: Optional[bool] = field(
    #     default=False,
    #     metadata={"help": "Enables using Huggingface auth token from Git Credentials."}
    # )

@dataclass
class DataArguments:
    force_remove_overlength_samples: bool = field(
        default=True,
        metadata={"help": "Remove overlength samples."}
    )
    eval_dataset_size: float = field(
        default=0.02, metadata={"help": "Ratio of dataset to use for validation."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    model_max_len: int = field(
        default=2048,
        metadata={"help": "Maximum model length (input and output).  Sequences will be right padded (and possibly truncated)."},
    )
    dataset: str = field(
        default='alpaca',
        metadata={"help": "Which dataset to finetune on. See datamodule for options."}
    )
    dataset_format: Optional[str] = field(
        default=None,
        metadata={"help": "Which dataset format is used. [alpaca|chip2|self-instruct|hh-rlhf|mistral]"}
    )

@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    task_name: str = field(
        default=None,
        metadata={"help": "The name of the task to train on."},
    )
    flash_attention: bool = field(
        default=True,
        metadata={"help": "Use flash attention."}
    ) 

    rerope: bool = field(
        default=False, 
        metadata={"help": "Use rerope."}
    )
    rerope_window: int = field(
        default=None,
        metadata={"help": "Rerope window size."}
    )

    neftune: bool = field(
        default=False,
        metadata={"help": "Use neftune."}
    )
    noise_alpha: float = field(
        default=5.0,
        metadata={"help": "Neftune noise alpha."}
    )

    wandb: str = field(
        default=None,
        metadata={"help": "Wandb project name."}
    )

    sample_packing: bool = field(
        default=False,
        metadata={"help": "Use sample packing for effiecient training."}
    )
    cache_dir: Optional[str] = field(
        default=None
    )
    full_finetune: bool = field(
        default=False,
        metadata={"help": "Finetune the entire model without adapters."}
    )
    adam8bit: bool = field(
        default=False,
        metadata={"help": "Use 8-bit adam."}
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=4,
        metadata={"help": "How many bits to use."}
    )
    lora_r: int = field(
        default=64,
        metadata={"help": "Lora R dimension."}
    )
    lora_alpha: float = field(
        default=16,
        metadata={"help": " Lora alpha."}
    )
    lora_dropout: float = field(
        default=0.0,
        metadata={"help":"Lora dropout."}
    )
    max_memory_MB: int = field(
        default=80000,
        metadata={"help": "Free memory per gpu."}
    )
    # report_to: str = field(
    #     default='none',
    #     metadata={"help": "To use wandb or something else for reporting."}
    # )
    mpt: bool = field(default=False, metadata={"help": 'Flag indicating whether this model is MPT or not'})
    output_dir: str = field(default='./output', metadata={"help": 'The output dir for logs and checkpoints'})
    optim: str = field(default='paged_adamw_32bit', metadata={"help": 'The optimizer to be used'})
    per_device_train_batch_size: int = field(default=1, metadata={"help": 'The training batch size per GPU. Increase for better speed.'})
    gradient_accumulation_steps: int = field(default=16, metadata={"help": 'How many gradients to accumulate before to perform an optimizer step'})
    num_train_epochs: int = field(default=3, metadata={"help": 'Number of training epochs.'})
    weight_decay: float = field(default=0.0, metadata={"help": 'The L2 weight decay rate of AdamW'}) # use lora dropout instead for regularization if needed
    learning_rate: float = field(default=0.0002, metadata={"help": 'The learning rate'})
    remove_unused_columns: bool = field(default=False, metadata={"help": 'Removed unused columns. Needed to make this codebase work.'})
    max_grad_norm: float = field(default=0.3, metadata={"help": 'Gradient clipping max norm. This is tuned and works well for all models tested.'})
    gradient_checkpointing: bool = field(default=False, metadata={"help": 'Use gradient checkpointing. You want to use this.'})
    do_train: bool = field(default=True, metadata={"help": 'To train or not to train, that is the question?'})
    lr_scheduler_type: str = field(default='constant', metadata={"help": 'Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis'})
    warmup_ratio: float = field(default=0.005, metadata={"help": 'Fraction of steps to do a warmup for'})
    logging_steps: int = field(default=10, metadata={"help": 'The frequency of update steps after which to log the loss'})
    group_by_length: bool = field(default=True, metadata={"help": 'Group sequences into batches with same length. Saves memory and speeds up training considerably.'})
    save_strategy: str = field(default='steps', metadata={"help": 'When to save checkpoints'})
    save_steps: int = field(default=250, metadata={"help": 'How often to save a model'})
    save_total_limit: int = field(default=1, metadata={"help": 'How many checkpoints to save before the oldest is overwritten'})
    deepspeed: str = field(default=None, metadata={"help": "deepspeed configuration path"})
    max_shard_size: str = field(default="5GB", metadata={"help": "Max shard size when saving model after full finetune."})

    repeat_steps: int = field(default=0, metadata={"help": "How many times to repeat the same batch."})

@dataclass
class GenerationArguments:
    # For more hyperparameters check:
    # https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
    # Length arguments
    max_new_tokens: Optional[int] = field(
        default=256,
        metadata={"help": "Maximum number of new tokens to be generated in evaluation or prediction loops"
                          "if predict_with_generate is set."}
    )
    min_new_tokens : Optional[int] = field(
        default=None,
        metadata={"help": "Minimum number of new tokens to generate."}
    )

    # Generation strategy
    do_sample: Optional[bool] = field(default=False)
    num_beams: Optional[int] = field(default=1)
    num_beam_groups: Optional[int] = field(default=1)
    penalty_alpha: Optional[float] = field(default=None)
    use_cache: Optional[bool] = field(default=True)

    # Hyperparameters for logit manipulation
    temperature: Optional[float] = field(default=1.0)
    top_k: Optional[int] = field(default=50)
    top_p: Optional[float] = field(default=1.0)
    typical_p: Optional[float] = field(default=1.0)
    diversity_penalty: Optional[float] = field(default=0.0)
    repetition_penalty: Optional[float] = field(default=1.0)
    length_penalty: Optional[float] = field(default=1.0)
    no_repeat_ngram_size: Optional[int] = field(default=0)

def find_all_linear_names(args, model):
    cls = bnb.nn.Linear4bit if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

class TrainHelperCallback(transformers.TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        clean_memory()

    def on_evaluate(self, args, state, control, **kwargs):
        clean_memory()

class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        logger.info('Saving PEFT checkpoint...')
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "adapter_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        def touch(fname, times=None):
            with open(fname, 'a'):
                os.utime(fname, times)

        touch(join(args.output_dir, 'completed'))
        self.save_model(args, state, kwargs)

def get_accelerate_model(args, checkpoint_dir):

    n_gpus = torch.cuda.device_count()
    max_memory = f'{args.max_memory_MB}MB'
    max_memory = {i: max_memory for i in range(n_gpus)}
    device_map = "auto"

    # if we are in a distributed setting, we need to set the device map and max memory per device
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': f'cuda:{local_rank}'}
        max_memory = {'': max_memory[local_rank]}


    if args.full_finetune: assert args.bits in [16, 32]

    logger.info(f'loading base model {args.model_name_or_path}...')

    config = transformers.AutoConfig.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
    )
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and args.model_max_len > orig_ctx_len:
        scaling_factor = float(math.ceil(args.model_max_len / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    compute_dtype = (torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))
    model_kwargs = {
        "cache_dir": args.cache_dir,
        "load_in_4bit": args.bits == 4,
        "load_in_8bit": args.bits == 8,
        "device_map": device_map if not args.deepspeed else None,
        "max_memory": max_memory if not args.deepspeed else None,
        "quantization_config": BitsAndBytesConfig(
            load_in_4bit=args.bits == 4,
            load_in_8bit=args.bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.double_quant,
            bnb_4bit_quant_type=args.quant_type,
        ) if args.bits in (4, 8) else None,
        "torch_dtype": (torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)),
        "trust_remote_code": args.trust_remote_code,
        "use_flash_attention_2": args.flash_attention,
        # "use_auth_token": args.use_auth_token
    }
    if args.mpt:
        model_kwargs["attn_config"] = {"attn_impl": "triton"}
    # model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, config=config, **model_kwargs)
    if compute_dtype == torch.float16 and args.bits == 4:
        major, minor = torch.cuda.get_device_capability()
        if major >= 8:
            print('='*80)
            print('Your GPU supports bfloat16, you can accelerate training with the argument --bf16')
            print('='*80)

    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)

    model.config.torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))

    if not args.full_finetune:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if not args.full_finetune:
        if checkpoint_dir is not None:
            logger.info("Loading adapters from checkpoint.")
            model = PeftModel.from_pretrained(model, join(checkpoint_dir, 'adapter_model'), is_trainable=True)
        else:
            logger.info(f'adding LoRA modules...')
            modules = find_all_linear_names(args, model)
            config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=modules,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, config)

    for name, module in model.named_modules():
        if "norm" in name:
            module.to(compute_dtype)
        if "lm_head" in name or "embed_tokens" in name:
            if hasattr(module, "weight"):
                module.to(compute_dtype)

    if args.neftune:
        from patches.neftune_monkey_patch import NEFTune
        model = NEFTune(model, noise_alpha=args.noise_alpha)

    return model

def print_trainable_parameters(args, model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    if args.bits == 4: trainable_params /= 2
    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable: {100 * trainable_params / all_param}"
    )

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

@dataclass
class DataCollatorForCausalLM(object):
    tokenizer: transformers.PreTrainedTokenizer
    model_max_len: int

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract elements
        sources = [f"{self.tokenizer.bos_token}{example['input']}" for example in instances]
        targets = [f"{example['output']}{self.tokenizer.eos_token}" for example in instances]

        # Tokenize
        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=self.model_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_targets = self.tokenizer(
            targets,
            max_length=self.model_max_len,
            truncation=True,
            add_special_tokens=False,
        )

        # Build the input and labels for causal LM
        input_ids = []
        labels = []
        for tokenized_source, tokenized_target in zip(
            tokenized_sources_with_prompt['input_ids'],
            tokenized_targets['input_ids']
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
            'attention_mask':input_ids.ne(self.tokenizer.pad_token_id),
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
    return {'input': prompt_format.format(**example)}

def local_dataset(dataset_name):
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
        return full_dataset.train_test_split(test_size=0.02, stratify_by_column='category')
    return full_dataset.train_test_split(test_size=0.02)

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
                    full_dataset = local_dataset(dataset_name)
                    return full_dataset
                except:
                    raise ValueError(f"Error loading dataset from {dataset_name}")
            else:
                raise NotImplementedError(f"Dataset {dataset_name} not implemented yet.")

    def format_dataset(dataset, dataset_format):
        if (
            dataset_format == 'alpaca' or dataset_format == 'alpaca-clean' or
            (dataset_format is None and args.dataset in ['alpaca', 'alpaca-clean'])
        ):
            dataset = dataset.map(extract_alpaca_dataset, remove_columns=['instruction'])
        elif dataset_format == 'chip2' or (dataset_format is None and args.dataset == 'chip2'):
            dataset = dataset.map(lambda x: {
                'input': x['text'].split('\n<bot>: ')[0].replace('<human>: ', ''),
                'output': x['text'].split('\n<bot>: ')[1],
            })
        elif dataset_format == 'self-instruct' or (dataset_format is None and args.dataset == 'self-instruct'):
            for old, new in [["prompt", "input"], ["completion", "output"]]:
                dataset = dataset.rename_column(old, new)
        elif dataset_format == 'hh-rlhf' or (dataset_format is None and args.dataset == 'hh-rlhf'):
            dataset = dataset.map(lambda x: {
                'input': '',
                'output': x['chosen']
            })
        elif dataset_format == 'oasst1' or (dataset_format is None and args.dataset == 'oasst1'):
            dataset = dataset.map(lambda x: {
                'input': '',
                'output': x['text'],
            })
        elif dataset_format == 'airoboros':
            logger.info("---------- Formatting dataset for Airoboros. ----------")
            def _format_airoboros(instruction):
                # FIXME - for Spider prompt
                if "### Instructions:" in instruction["instruction"]:
                    in_ = instruction["instruction"]
                    out_ = instruction['response']
                    return {
                        'input': in_,
                        'output': out_,
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
                        'input': in_,
                        'output': instruction['response'].strip() + "\n",
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
                    return {
                        'input': in_,
                        'output': out_,
                    }
                else:
                    in_ = f"<s>[INST] {instruction['instruction']} [/INST]"
                    out_ = f"{instruction['response']}</s>"
                    return {
                        'input': in_,
                        'output': out_,
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
                    'input': in_,
                    'output': out_,
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
                    
                return {'input': in_,
                        'output': out_}

            dataset = dataset.map(_format_instruction_input_response)

        elif dataset_format == 'input-output':
            # leave as is
            pass
            def _format_input_output(instruction):
                return {
                    'input': instruction['instruction'],
                    'output': instruction['response'],
                }
            dataset = dataset.map(_format_input_output)

        # Remove unused columns.
        dataset = dataset.remove_columns(
            [col for col in dataset.column_names['train'] if col not in ['input', 'output']]
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
        if args.max_train_samples is not None and len(train_dataset) > args.max_train_samples:
            train_dataset = train_dataset.select(range(args.max_train_samples))
        if args.group_by_length:
            train_dataset = train_dataset.map(lambda x: {'length': len(x['input']) + len(x['output'])})
            
    # Remove any training data that exceeds the max length.
    def _get_data_length(item):
        prompt = f"{tokenizer.bos_token}{item['input']}{item['output']}{tokenizer.eos_token}"
        return len(
            tokenizer(
                prompt,
                max_length=args.model_max_len + 1,
                truncation=True,
                add_special_tokens=False
            ).input_ids
        )
    if args.force_remove_overlength_samples:
        logger.info(f"---------- Filtering out samples longer than {args.model_max_len} ----------")  
        prev_len = len(train_dataset)
        train_dataset = train_dataset.filter(
            lambda x: _get_data_length(x) < args.model_max_len - 10
        )
        logger.info(f"Filtered out {prev_len - len(train_dataset)} samples. ({len(train_dataset)}/{prev_len})")

    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        model_max_len=args.model_max_len,
    )

    # FIXME
    if args.repeat_steps > 0:
        one_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps * torch.cuda.device_count()
        train_dataset = RepeatDataset(train_dataset, repeat_batch_size= one_batch_size, repeat_steps = args.repeat_steps)
    
    return dict(
        train_dataset=train_dataset if args.do_train else None,
        eval_dataset=eval_dataset if args.do_eval else None,
        predict_dataset=eval_dataset if args.do_predict else None,
        data_collator=data_collator
    )

def get_last_checkpoint(checkpoint_dir):
    if isdir(checkpoint_dir):
        is_completed = exists(join(checkpoint_dir, 'completed'))
        if is_completed: return None, True # already finished
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if isdir(join(checkpoint_dir, filename)) and filename.startswith('checkpoint'):
                max_step = max(max_step, int(filename.replace('checkpoint-', '')))
        if max_step == 0: return None, is_completed # training started, but no checkpoint
        checkpoint_dir = join(checkpoint_dir, f'checkpoint-{max_step}')
        logger.info(f"Found a previous checkpoint at: {checkpoint_dir}")
        return checkpoint_dir, is_completed # checkpoint found!
    return None, False # first training

# from peft.utils.other import WEIGHTS_NAME
# from transformers.trainer import TRAINING_ARGS_NAME
# from transformers.modeling_utils import unwrap_model

# def get_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]: # get state dict containing trainable parameters
#     state_dict = model.state_dict()
#     filtered_state_dict = {}

#     for k, v in model.named_parameters():
#         if v.requires_grad:
#             filtered_state_dict[k] = state_dict[k].cpu().clone().detach()

#     return filtered_state_dict

# class PeftTrainer(Seq2SeqTrainer):
#     r"""
#     Inherits Seq2SeqTrainer to support parameter-efficient checkpoints.
#     """

#     # def __init__(self, finetuning_args: FinetuningArguments, **kwargs):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         if self.is_world_process_zero() and os.path.exists(os.path.join(self.args.output_dir, "trainer_log.jsonl")):
#             logger.warning("Previous log file in this folder will be deleted.")
#             os.remove(os.path.join(self.args.output_dir, "trainer_log.jsonl"))

#     def _save(self, output_dir: Optional[str] = None, state_dict: Optional[Dict[str, torch.Tensor]] = None) -> None:
#         r"""
#         Saves trainable parameters as model checkpoint.

#         This function will only be executed at the process zero.

#         Subclass and override to inject custom behavior. It should not be directly used by external scripts.
#         """
#         output_dir = output_dir if output_dir is not None else self.args.output_dir
#         os.makedirs(output_dir, exist_ok=True)
#         logger.info(f"Saving model checkpoint to {output_dir}")
#         model = unwrap_model(self.model)

#         if hasattr(model, "pretrained_model"): # for models with valuehead
#             backbone_model = getattr(model, "pretrained_model")
#         else:
#             backbone_model = model

#         if hasattr(backbone_model, "peft_config"): # peft methods
#             backbone_model.save_pretrained(output_dir, state_dict=get_state_dict(backbone_model)) # save lora weights
#         else:
#             torch.save(get_state_dict(backbone_model), os.path.join(output_dir, WEIGHTS_NAME)) # save trainable weights

#         # if hasattr(model, "v_head"): # save valuehead weights
#         #     torch.save(get_state_dict(getattr(model, "v_head")), os.path.join(output_dir, VALUE_HEAD_FILE_NAME))

#         with open(os.path.join(output_dir, TRAINING_ARGS_NAME), "w", encoding="utf-8") as f:
#             f.write(self.args.to_json_string() + "\n")
#         # self.finetuning_args.save_to_json(os.path.join(output_dir, FINETUNING_ARGS_NAME))

#     # def _load_best_model(self):
#     #     r"""
#     #     Loads trainable parameters from model checkpoint.

#     #     Subclass and override to inject custom behavior. It should not be directly used by external scripts.
#     #     """
#     #     logger.info(f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric}).")
#     #     model = unwrap_model(self.model)
#     #     if hasattr(model, "peft_config"): # peft methods
#     #         model.load_adapter(self.state.best_model_checkpoint, getattr(model, "active_adapter"))
#     #     else:
#     #         load_trainable_params(model, self.state.best_model_checkpoint)

#     #     if hasattr(model, "v_head"):
#     #         load_valuehead_params(model, self.state.best_model_checkpoint)

def setup_wandb(args):
    # Check if parameter passed or if set within environ
    wandb_project = args.wandb
    if wandb_project is None:
        return

    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    # if len(wandb_watch) > 0:
    #     os.environ["WANDB_WATCH"] = wandb_watch
    # if len(wandb_log_model) > 0:
    #     os.environ["WANDB_LOG_MODEL"] = wandb_log_model

from transformers import SchedulerType
from transformers.optimization import TYPE_TO_SCHEDULER_FUNCTION
def get_lr_scheduler(lr_scheduler_type, optimizer, num_warmpup_steps, num_training_steps, optim_args):
    name = SchedulerType(lr_scheduler_type)
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]
    optim_args = optim_args or {}
    return schedule_func(optimizer, num_warmup_steps=num_warmpup_steps, num_training_steps=num_training_steps, **optim_args)

def train():
    hfparser = transformers.HfArgumentParser((
        ModelArguments, DataArguments, TrainingArguments, GenerationArguments
    ))
    model_args, data_args, training_args, generation_args, extra_args = \
        hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    # training_args.generation_config = transformers.GenerationConfig(**vars(generation_args))
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )

    if args.task_name is None:
        args.task_name = os.path.basename(os.curdir)
        
    from datetime import datetime
    logger.add(f"{args.output_dir}/logs/finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log", level="INFO")
    logger.info(f"{args=}")

    # setup_wandb(args)

    # if args.flash_attention:
    #     from patches.flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
    #     replace_llama_attn_with_flash_attn(packed=args.sample_packing)
    #     logger.info(f"Enabled flash attention monkey patching.")

    if args.rerope:
        from patches.rerope_monkey_patch import replace_llama_attention_forword_with_rerope
        rerope_window = args.rerope_window or int(args.model_max_Len * 3 / 8) // 16 * 16
        replace_llama_attention_forword_with_rerope(training_length=args.model_max_len, window=rerope_window)
        logger.info(f"Enabled rerope monkey patching.")

    checkpoint_dir, completed_training = get_last_checkpoint(args.output_dir)
    if completed_training:
        logger.warning('Detected that training was already completed!')

    model = get_accelerate_model(args, checkpoint_dir)

    model.config.use_cache = False
    if not args.deepspeed:
        print_trainable_parameters(args, model)
    logger.info('loaded model')
    set_seed(args.seed)

    # Tokenizer
    tokenizer_kwargs = {
        "cache_dir": args.cache_dir,
        "padding_side": "left",
        "use_fast": False,
    }
    # if args.mpt:
    #     tokenizer_kwargs["padding_side"] = "left"
    #     tokenizer_kwargs.pop("use_fast")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, **tokenizer_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0
    data_module = make_data_module(tokenizer=tokenizer, args=args)
    trainer = Seq2SeqTrainer(
    # trainer = PeftTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **{k:v for k,v in data_module.items() if k != 'predict_dataset'},
    )

    # Callbacks
    if not args.full_finetune:
        trainer.add_callback(SavePeftModelCallback)

    trainer.add_callback(TrainHelperCallback)

    # Verifying the datatypes.
    if not args.full_finetune:
        dtypes = {}
        for _, p in model.named_parameters():
            dtype = p.dtype
            if dtype not in dtypes: dtypes[dtype] = 0
            dtypes[dtype] += p.numel()
        total = 0
        for k, v in dtypes.items(): total+= v
        for k, v in dtypes.items():
            print(k, v, v/total)

    all_metrics = {"run_name": args.run_name}
    # Training
    if args.do_train:
        logger.info("*** Train ***")
        # Note: `resume_from_checkpoint` not supported for adapter checkpoints by HF.
        # Currently adapter checkpoint is reloaded as expected but optimizer/scheduler states are not.
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        all_metrics.update(metrics)
    # Evaluation
    if args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        all_metrics.update(metrics)
    # Prediction
    if args.do_predict:
        logger.info("*** Predict ***")
        prediction_output = trainer.predict(test_dataset=data_module['predict_dataset'],metric_key_prefix="predict")
        prediction_metrics = prediction_output.metrics
        predictions = prediction_output.predictions
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        predictions = tokenizer.batch_decode(
            predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        with open(os.path.join(args.output_dir, 'predictions.jsonl'), 'w') as fout:
            for i, example in enumerate(data_module['predict_dataset']):
                example['prediction_with_input'] = predictions[i].strip()
                example['prediction'] = predictions[i].replace(example['input'], '').strip()
                fout.write(json.dumps(example) + '\n')
        print(prediction_metrics)
        trainer.log_metrics("predict", prediction_metrics)
        trainer.save_metrics("predict", prediction_metrics)
        all_metrics.update(prediction_metrics)

    if (args.do_train or args.do_eval or args.do_predict):
        with open(os.path.join(args.output_dir, "metrics.json"), "w") as fout:
            fout.write(json.dumps(all_metrics))

    # Safely save final full-tune model.
    if args.full_finetune:
        state_dict = trainer.model.state_dict()
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        trainer.model.save_pretrained(args.output_dir, state_dict=cpu_state_dict, max_shard_size=args.max_shard_size)
        tokenizer.save_pretrained(args.output_dir)
        with open(os.path.join(args.output_dir, "config.json")) as infile:
            config = json.loads(infile.read())
        config["_name_or_path"] = os.path.basename(args.output_dir)
        with open(os.path.join(args.output_dir, "config.json"), "w") as outfile:
            outfile.write(json.dumps(config, indent=2))


if __name__ == "__main__":
    train()
