#!/usr/bin/env python
# coding: utf-8
"""
dist_train:
	export CUDA_VISIBLE_DEVICES=0,1,2,3,4 \
	accelerate launch --num_processes 3 \
		--config_file ./deepspeed_zero3.yaml \
		r1_finetune.py \
		--config ./grpo-qwen-2.5-3b-deepseek-r1-countdown.yaml

train:
	python r1_finetune.py \
		--config ./grpo-qwen-2.5-3b-deepseek-r1-countdown.yaml
"""

import os, sys
import re
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Union, List, Dict, Any

import torch
from transformers.trainer_utils import get_last_checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers import (
    set_seed,
    BitsAndBytesConfig,
)

from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel
)
import bitsandbytes as bnb
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer, get_peft_config, ModelConfig, TrlParser
from tqdm import tqdm
from loguru import logger

from speechless.finetune.callbacks import LoggingCallback, CleanMemoryCallback, SavePeftModelCallback


@dataclass
class CustomArguments:
    dataset_id_or_path: str = None
    dataset_splits: str = "train"
    tokenizer_name_or_path: str = None
    loader_type: str = "huggingface"  # unsloth or huggingface
    test_size: float = 100
    shuffle_seed: int = 10042

    reward_functions: str = ""
    dataset_map_functions: str = ""


import gc, ctypes


def clean_memory(num_refresh=3):
    for _ in range(num_refresh):
        gc.collect()
        if sys.platform == 'linux':
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        # mps backend
        if torch.backends.mps.is_available():
            torch.cuda.empty_cache()


# -------------------- Model --------------------
# Load model by using unsloth
def load_model_and_tokenizer_unsloth(
    model_path: str,
    lora_path: str = None,
    lora_config: LoraConfig = None,
    max_seq_length: int = 8192,
):
    # FIXME - hardcoded values
    lora_rank = 64

    from unsloth import FastLanguageModel, PatchFastRL
    PatchFastRL("GRPO", FastLanguageModel)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        load_in_4bit=True,  # False for LoRA 16bit
        fast_inference=True,  # Enable vLLM fast inference
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.5,  # Reduce if out of memory
    )

    if lora_path is not None:
        logger.info("Loading adapters from checkpoint.")
        model = PeftModel.from_pretrained(model, lora_path, is_trainable=True)
    else:
        if lora_config is None:
            lora_config = LoraConfig(
                r=lora_rank,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
                lora_alpha=lora_rank,
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ], # Remove QKVO if out of memory
            )
        model = FastLanguageModel.get_peft_model(
            model,
            use_gradient_checkpointing="unsloth",  # Enable long context finetuning
            random_state=10042,
            **lora_config.__dict__
        )

    model.config.use_cache = False  # Disables KV caching to save memory.
    # Then enable gradient checkpointing
    model.gradient_checkpointing_enable()

    return model, tokenizer


def load_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


SUPPORTS_BFLOAT16 = False
if torch.cuda.is_available():
    major_version, minor_version = torch.cuda.get_device_capability()
    if major_version >= 8:
        SUPPORTS_BFLOAT16 = True
def is_bfloat16_supported():
    return SUPPORTS_BFLOAT16

# Load model by using Huggingface
def load_model_huggingface(model_path: str, lora_path: str = None, lora_config: LoraConfig = None):

    compute_dtype = torch.bfloat16 if is_bfloat16_supported() else torch.float16
    bnb_config = None
    if torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            load_in_8bit=False,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ) 
    model_kwargs = {
        # "cache_dir": args.cache_dir,
        # transformers-4.39.0.dev0
        # ValueError: You can't pass `load_in_4bit`or `load_in_8bit` as a kwarg when passing `quantization_config` argument at the same time.
        # "load_in_4bit": args.bits == 4,
        # "load_in_8bit": args.bits == 8,
        # "device_map": "auto", # deepspeed  not use "auto"
        "quantization_config": bnb_config,
        "torch_dtype": compute_dtype,
        "trust_remote_code": True,
        # "use_flash_attention_2": args.flash_attention,
        # "use_auth_token": args.use_auth_token
    }

    if torch.cuda.is_available():
        model_kwargs["attn_implementation"] = "flash_attention_2"

    # if args.mpt:
    #     model_kwargs["attn_config"] = {"attn_impl": "triton"}

    logger.info(f"{model_kwargs=}")
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

    def find_all_linear_names(model, bits: int = 4):
        cls = bnb.nn.Linear4bit if bits == 4 else (bnb.nn.Linear8bitLt if bits == 8 else torch.nn.Linear)
        lora_module_names = set()
        for name, module in model.named_modules():
            if isinstance(module, cls):
                names = name.split('.')
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])

        if 'lm_head' in lora_module_names:  # needed for 16-bit
            lora_module_names.remove('lm_head')
        return list(lora_module_names)


    if lora_path is not None:
        logger.info("Loading adapters from checkpoint.")
        model = PeftModel.from_pretrained(model, lora_path, is_trainable=True)
    else:

        if lora_config is None:
            lora_config = LoraConfig(
                r=64,
                lora_alpha=128,
            )
        if lora_config.target_modules is None:
            logger.info(f'adding LoRA modules...')
            all_modules = find_all_linear_names(model)
            print(f"LoRA modules: {all_modules}")
            # target_modules=[
            #     "q_proj",
            #     "k_proj",
            #     "v_proj",
            #     "o_proj",
            #     "gate_proj",
            #     "up_proj",
            #     "down_proj",
            # ]
            target_modules = all_modules
            # Remove QKVO if out of memory
            lora_config.target_modules = target_modules

        model = get_peft_model(model, lora_config)

    return model


def load_model_and_tokenizer_huggingface(
    model_path: str,
    lora_path: str = None,
    lora_config: LoraConfig = None,
    tokenizer_name_or_path: str = None,
):
    tokenizer = load_tokenizer(tokenizer_name_or_path if tokenizer_name_or_path else model_path)
    model = load_model_huggingface(model_path, lora_path=lora_path, lora_config=lora_config)

    model.config.use_cache = False  # Disables KV caching to save memory.
    # Then enable gradient checkpointing
    model.gradient_checkpointing_enable()

    return model, tokenizer


def load_model_and_tokenizer(
    model_path: str,
    lora_path: str = None,
    lora_params: dict = None,
    tokenizer_name_or_path: str = None,
    loader_type: str = "huggingface",
    max_seq_length: int = 8192,
):
    """
    Load model and tokenizer from either unsloth or huggingface.

    """

    lora_params = lora_params or {
        "r": 64,
        "lora_alpha": 128
    }
    lora_config = LoraConfig(**lora_params)

    if loader_type == "unsloth":
        return load_model_and_tokenizer_unsloth(
            model_path,
            lora_path=lora_path,
            lora_config=lora_config,
            max_seq_length=max_seq_length,
        )
    elif loader_type == "huggingface":
        return load_model_and_tokenizer_huggingface(
            model_path,
            lora_path=lora_path,
            lora_config=lora_config,
            tokenizer_name_or_path=tokenizer_name_or_path,
        )
    else:
        raise Exception(f"Unknown loader_type: {loader_type} in load_model_and_tokenizer.")


def build_datasets(
    dataset_id_or_path,
    tokenizer,
    dataset_splits="train",
    dataset_map_function=None,
    test_size=100,
    shuffle_seed=10042
):
    logger.info(f"Loading dataset {dataset_id_or_path} with splits {dataset_splits}")

    if isinstance(dataset_id_or_path, str):
        datasets_list = dataset_id_or_path.split(",")
    elif isinstance(dataset_id_or_path, list):
        datasets_list = [dataset_id_or_path]
    else:
        raise ValueError(f"dataset_id_or_path must be a string or a list of strings, got {dataset_id_or_path}")

    dataset = load_dataset(datasets_list[0], split=dataset_splits)
    for dataset_id in dataset_id_or_path[1:]:
        dataset = dataset.concatenate(load_dataset(dataset_id, split=dataset_splits))

    dataset = dataset.shuffle(seed=shuffle_seed)

    if dataset_map_function is not None:
        dataset = dataset_map_function(dataset, tokenizer=tokenizer)

    train_test_split = dataset.train_test_split(test_size=test_size)
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]

    return train_dataset, eval_dataset



# -------------------- Training Callbacks --------------------

def r1_finetune(training_args: GRPOConfig, model_args: ModelConfig, custom_args: CustomArguments, r1_params: dict):
    logger.info(f"Training parameters: {training_args}")
    logger.info(f"Model parameters: {model_args}")
    logger.info(f"Custom parameters: {custom_args}")

    reward_functions = r1_params["reward_functions"]
    dataset_map_functions = r1_params["dataset_map_functions"]

    model_path = model_args.model_name_or_path
    dataset_id_or_path = custom_args.dataset_id_or_path
    loader_type = custom_args.loader_type  # unsloth or huggingface
    test_size = custom_args.test_size
    shuffle_seed = custom_args.shuffle_seed

    # -------------------- Model and Tokenizer --------------------
    lora_params = {
        "r": model_args.lora_r,
        "lora_alpha": model_args.lora_alpha,
        "target_modules": model_args.lora_target_modules,
    }
    model, tokenizer = load_model_and_tokenizer(model_path, loader_type=loader_type, lora_params=lora_params)

    # -------------------- Dataset --------------------
    train_dataset, eval_dataset = build_datasets(
        dataset_id_or_path, tokenizer=tokenizer, dataset_map_function=dataset_map_functions, test_size=test_size, shuffle_seed=shuffle_seed
    )

    # -------------------- Training loop --------------------
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_functions,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_args),
    )

    trainer.add_callback(SavePeftModelCallback)
    trainer.add_callback(CleanMemoryCallback)
    trainer.add_callback(LoggingCallback)

    # Train the model
    logger.info(
        f'*** Starting training {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} for {training_args.num_train_epochs} epochs***'
    )

    # Check for last checkpoint
    last_checkpoint = None
    # last_checkpoint = get_checkpoint(training_args)
    # if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
    #     logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

    # -------------------- Save model --------------------
    def save_model(trainer, training_args):

        # Log and save metrics
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        # Save model
        logger.info("*** Save model ***")
        trainer.model.config.use_cache = True
        trainer.save_model(training_args.output_dir)
        logger.info(f"Model saved to {training_args.output_dir}")
        training_args.distributed_state.wait_for_everyone()  # wait for all processes to load

        # Save tokenizer
        tokenizer.save_pretrained(training_args.output_dir)
        logger.info(f"Tokenizer saved to {training_args.output_dir}")

        # Save everything else on main process
        if trainer.accelerator.is_main_process:
            trainer.create_model_card({
                "tags": ["rl", "grpo", "r1"],
            })

        # push to hub if needed
        if training_args.push_to_hub is True:
            logger.info("Pushing to hub...")
            trainer.push_to_hub()

    save_model(trainer, training_args)

    logger.info("*** Training complete! ***")


def main():
    parser = TrlParser((GRPOConfig, ModelConfig, CustomArguments))
    training_args, model_args, custom_args  = parser.parse_args_and_config()

    r1_params = {
        "reward_functions": [],
        "dataset_map_functions": []
    }
    # Run the main training loop
    r1_finetune(training_args, model_args, custom_args, r1_params=r1_params)


if __name__ == "__main__":
    main()
