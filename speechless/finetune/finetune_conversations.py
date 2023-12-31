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

# from speechless.patches.llama_attn_replace_sft import replace_llama_attn
# replace_llama_attn(use_flash_attn=True, use_full=False, inference=False)

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

def qwen_prepare_model_for_kbit_training(model, use_gradient_checkpointing=True):
    """
    This method wraps the entire protocol for preparing a model before running a training. This includes:
        1- Cast the layernorm in fp32 2- making output embedding layer require grads 3- Add the upcasting of the lm
        head to fp32

    Args:
        model, (`transformers.PreTrainedModel`):
            The loaded model from `transformers`
    """
    loaded_in_kbit = getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False)
    is_gptq_quantized = getattr(model, "quantization_method", None) == "gptq"
    for name, param in model.named_parameters():
        # freeze base model's layers
        param.requires_grad = False

    # if not is_gptq_quantized:
    #     # cast all non INT8 parameters to fp32
    #     for param in model.parameters():
    #         if (param.dtype == torch.float16) or (param.dtype == torch.bfloat16):
    #             param.data = param.data.to(torch.float32)

    if (loaded_in_kbit or is_gptq_quantized) and use_gradient_checkpointing:
        # For backward compatibility
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        # enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()

    return model



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
        default=True,
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
    max_train_samples: Optional[float] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set. If set to a float, will truncate the number of examples to that percentage of the dataset."
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
        default="conversations",
        metadata={"help": "Which dataset format is used. [alpaca|conversations|chip2|self-instruct|hh-rlhf|mistral]"}
    )
    prompt_type: Optional[str] = field(
        default=None,
        metadata={"help": "Which prompt type to use. [alpaca|chatlm|conversations|chip2|self-instruct|hh-rlhf|mistral]"}
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

    long_lora: bool = field(
        default=False,
        metadata={"help": "Use long lora."}
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


    sliding_window: int = field(
        default=4096,
        metadata={"help": "Sliding window size."}
    )

    rope_theta: float = field(
        default=10000,
        metadata={"help": "Rope theta."}
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

class CleanMemoryCallback(transformers.TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        clean_memory()

    def on_evaluate(self, args, state, control, **kwargs):
        clean_memory()

from transformers import TrainerCallback
class LoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            if "eval_loss" in logs:
                logger.info(logs)
            else:
                logger.debug(logs)

class SavePeftModelCallback(TrainerCallback):
    def save_model(self, args, state, kwargs):
        logger.info('Saving PEFT checkpoint...')
        # if state.best_model_checkpoint is not None:
        #     checkpoint_folder = os.path.join(state.best_model_checkpoint, "adapter_model")
        # else:
        #     checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)

        self._symlink_latest_checkpoint(checkpoint_folder)

    def _symlink_latest_checkpoint(self, checkpoint_folder):
        # if the latest checkpoint is a symlink, remove it
        output_dir = os.path.dirname(checkpoint_folder)
        latest_checkpoint = os.path.join(output_dir, "latest")
        if os.path.islink(latest_checkpoint):
            os.remove(latest_checkpoint)
        # symlink the latest checkpoint to the checkpoint folder
        os.symlink(os.path.basename(checkpoint_folder), latest_checkpoint)

    def on_save(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        def touch(fname, times=None):
            with open(fname, 'a'):
                os.utime(fname, times)

        if state.is_local_process_zero:
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
        trust_remote_code=args.trust_remote_code,
    )
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and args.model_max_len > orig_ctx_len:
        scaling_factor = float(math.ceil(args.model_max_len / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    config.sliding_window = args.sliding_window
    config.rope_theta = args.rope_theta


    if "qwen" in args.model_name_or_path.lower():
        if args.bf16:
            config.bf16 = True
        elif args.fp16:
            config.fp16 = True

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
        "torch_dtype": (torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)),
        "trust_remote_code": args.trust_remote_code,
        "use_flash_attention_2": args.flash_attention,
        # "use_auth_token": args.use_auth_token
    }

    # if args.mpt:
    #     model_kwargs["attn_config"] = {"attn_impl": "triton"}

    logger.info(f"{model_kwargs=}")
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

    model.config.torch_dtype=(torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))

    if not args.full_finetune:

        if "qwen" in args.model_name_or_path.lower():
            model = qwen_prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
        else:
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
            print(f"LoRA modules: {modules}")
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
        from speechless.patches.neftune_monkey_patch import NEFTune
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
    model_max_len: int,
    tokenizer: transformers.PreTrainedTokenizer,
    template: str="tool-llama-single-round"
) -> Dict:
    # conv = get_conversation_template(template)
    # if template == "tool-llama":
    #     roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    # elif template == "tool-llama-single-round" or template == "tool-llama-multi-rounds":
    #     roles = {"system": conv.roles[0], "user": conv.roles[1], "function": conv.roles[2], "assistant": conv.roles[3]}

    roles = {"system": "System", "user": "User", "function": "Function", "assistant": "Assistant"}
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
        max_length=model_max_len,
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
        labels[cur_len : cur_len + instruction_len] = IGNORE_INDEX
        cur_len += turn_len

    labels[cur_len:] = IGNORE_INDEX

    # if False:  # Inspect and check the correctness of masking
    #     z = labels.clone()
    #     z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
    #     rank0_print(tokenizer.decode(z))

    if cur_len < model_max_len:
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
    model_max_len: int,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # conv = get_conversation_template(template)
    # if template == "tool-llama":
    #     roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    # elif template == "tool-llama-single-round" or template == "tool-llama-multi-rounds":
    #     roles = {"system": conv.roles[0], "user": conv.roles[1], "function": conv.roles[2], "assistant": conv.roles[3]}

    roles = {"system": "System", "user": "User", "function": "Function", "assistant": "Assistant"}
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

        tokenized_source = tokenizer(source, max_length=model_max_len, truncation=True, add_special_tokens=False)
        tokenized_target = tokenizer(target, max_length=model_max_len, truncation=True, add_special_tokens=False)
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

@dataclass
class DialogDataCollatorForCausalLM(object):
    tokenizer: transformers.PreTrainedTokenizer
    model_max_len: int
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
                data_dict = preprocess_toolbench_dataset(example,
                                                   model_max_len=self.model_max_len,
                                                   tokenizer=self.tokenizer,
                                                   template="tool-llama-single-round")
                example_input_ids = data_dict['input_ids']
                example_labels = data_dict['labels']
                if example_input_ids is not None:
                    # print(f"{example_input_ids.shape=},{example_input_ids=}")
                    # print(f"{example_labels.shape=},{example_labels=}")
                    input_ids.append(example_input_ids)
                    labels.append(example_labels)
                continue

            elif prompt_type == "tool-llama-multi-rounds":
                data_dict = preprocess_multi_rounds_dialog(example,
                                                   model_max_len=self.model_max_len,
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
                else: # default alpaca
                    source, target = generate_round_prompt_alpaca(
                        idx,
                        human_input,
                        bot_response,
                        bos_token=self.tokenizer.bos_token,
                        eos_token=self.tokenizer.eos_token,
                        system_prompt=system_prompt,
                    )

                tokenized_source = self.tokenizer(source,
                                                  max_length=self.model_max_len,
                                                  truncation=True,
                                                  add_special_tokens=False,
                                                  )
                tokenized_target = self.tokenizer(target,
                                                  max_length=self.model_max_len,
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
            dataset = dataset.map(lambda x: {
                'conversations': [(x['input'], x['output'])]
            })

        elif dataset_format == 'chip2' or (dataset_format is None and args.dataset == 'chip2'):
            # dataset = dataset.map(lambda x: {
            #     'input': x['text'].split('\n<bot>: ')[0].replace('<human>: ', ''),
            #     'output': x['text'].split('\n<bot>: ')[1],
            # })
            dataset = dataset.map(lambda x: {
                'conversations': [(x['text'].split('\n<bot>: ')[0].replace('<human>: ', ''), x['text'].split('\n<bot>: ')[1])]
            })
        elif dataset_format == 'self-instruct' or (dataset_format is None and args.dataset == 'self-instruct'):
            # for old, new in [["prompt", "input"], ["completion", "output"]]:
            #     dataset = dataset.rename_column(old, new)
            dataset = dataset.map(lambda x: {
                'conversations': [(x['prompt'], x['completion'])]
            })
        elif dataset_format == 'hh-rlhf' or (dataset_format is None and args.dataset == 'hh-rlhf'):
            # dataset = dataset.map(lambda x: {
            #     'input': '',
            #     'output': x['chosen']
            # })
            dataset = dataset.map(lambda x: {
                'conversations': [('', x['chosen'])]
            })
        elif dataset_format == 'oasst1' or (dataset_format is None and args.dataset == 'oasst1'):
            # dataset = dataset.map(lambda x: {
            #     'input': '',
            #     'output': x['text'],
            # })
            dataset = dataset.map(lambda x: {
                'conversations': [('', x['text'])]
            })
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
                    return {'conversations': [(in_, out_)]}
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
                    return {'conversations': [(in_, instruction['response'].strip() + "\n")]}

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
                    return {'conversations': [(in_, out_)]}
                else:
                    in_ = f"<s>[INST] {instruction['instruction']} [/INST]"
                    out_ = f"{instruction['response']}</s>"
                    # return {
                    #     'input': in_,
                    #     'output': out_,
                    # }
                    return {'conversations': [(in_, out_)]}
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
                return {'conversations': [(in_, out_)]}

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
                return {'conversations': [(in_, out_)]}

            dataset = dataset.map(_format_instruction_input_response)

        elif dataset_format == 'input-output':
            # leave as is
            pass
            def _format_input_output(instruction):
                # return {
                #     'input': instruction['instruction'],
                #     'output': instruction['response'],
                # }
                return {'conversations': [(instruction['instruction'], instruction['response'])]}

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
                return {'conversations': human_bot_dialog}

            dataset = dataset.map(_format_multi_turns)

        # Remove unused columns.
        dataset = dataset.remove_columns(
            # FIXME
            # [col for col in dataset.column_names['train'] if col not in ['input', 'output']]
            [col for col in dataset.column_names['train'] if col not in ['conversations', 'system_prompt', 'prompt_type']]
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

    # FIXME
    # data_collator = DataCollatorForCausalLM(
    data_collator = DialogDataCollatorForCausalLM(
        tokenizer=tokenizer,
        model_max_len=args.model_max_len,
        prompt_type=args.prompt_type,
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
    logger.add(f"{args.output_dir}/logs/{args.task_name}-{datetime.now().strftime('%Y%m%d_%H%M%S')}.log", level="INFO")
    logger.info(f"{args=}")

    # setup_wandb(args)

    # if args.flash_attention:
    #     from speechless.patches.flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
    #     replace_llama_attn_with_flash_attn(packed=args.sample_packing)
    #     logger.info(f"Enabled flash attention monkey patching.")

    if args.long_lora:
        if 'mistral' in args.model_name_or_path:
            logger.warning(f"Mistral doesn't support long alpaca now.")
            # from speechless.patches.llama_attn_replace_sft import replace_mistral_attn
            # replace_mistral_attn(use_flash_attn=training_args.flash_attention, use_full=False, inference=False)
        else:
            from speechless.patches.long_lora_monkey_patch import replace_llama_attn
            replace_llama_attn(use_flash_attn=training_args.flash_attention, use_full=False, inference=False)

    if args.rerope:
        from speechless.patches.rerope_monkey_patch import replace_llama_attention_forword_with_rerope
        rerope_window = args.rerope_window or int(args.model_max_Len * 3 / 8) // 16 * 16
        replace_llama_attention_forword_with_rerope(training_length=args.model_max_len, window=rerope_window)
        logger.info(f"Enabled rerope monkey patching.")

    # if args.sliding_window > 0:
    #     if 'mistral' not in args.model_name_or_path:
    #         from speechless.patches.sliding_window_monkey_patch import replace_llama_attn
    #         replace_llama_attn()

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
        "trust_remote_code": True,
    }
    # if args.mpt:
    #     tokenizer_kwargs["padding_side"] = "left"
    #     tokenizer_kwargs.pop("use_fast")
    print(f"---------- Original tokens----------")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, **tokenizer_kwargs)
    print(f"{tokenizer.pad_token=},{tokenizer.pad_token_id=}")
    print(f"{tokenizer.unk_token=},{tokenizer.unk_token_id=}")
    print(f"{tokenizer.bos_token=},{tokenizer.bos_token_id=}")
    print(f"{tokenizer.eos_token=},{tokenizer.eos_token_id=}")

    if "qwen" in args.model_name_or_path.lower():
        tokenizer.eos_token = "<|endoftext|>"
        # tokenizer.unk_token = "<|extra_3|>"
        tokenizer.bos_token = "<|extra_2|>"
        tokenizer.pad_token = "<|extra_1|>"
    else:
        if tokenizer.bos_token_id is None:
            tokenizer.bos_token_id = 1
            tokenizer.bos_token = "<s>"
        if tokenizer.eos_token_id is None:
            tokenizer.eos_token_id = 2
            tokenizer.eos_token = "</s>"
        # if tokenizer.unk_token_id is None:
        #     tokenizer.unk_token_id = 0
        #     tokenizer.unk_token = "<unk>"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = 0 # tokenizer.eos_token_id
            tokenizer.pad_token = tokenizer._convert_id_to_token(tokenizer.pad_token_id) #tokenizer.eos_token
    print(f"---------- Fixed tokens ----------")
    print(f"{tokenizer.pad_token=},{tokenizer.pad_token_id=}")
    # print(f"{tokenizer.unk_token=},{tokenizer.unk_token_id=}")
    print(f"{tokenizer.bos_token=},{tokenizer.bos_token_id=}")
    print(f"{tokenizer.eos_token=},{tokenizer.eos_token_id=}")
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

    trainer.add_callback(CleanMemoryCallback)
    trainer.add_callback(LoggingCallback)


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
        predictions = np.where(predictions != IGNORE_INDEX, predictions, tokenizer.pad_token_id)
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
