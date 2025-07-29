#!/usr/bin/env python
import os, json, re
import random
import torch
from copy import deepcopy
from loguru import logger
from tqdm import tqdm
from typing import Any
# Unsloth shoud be imported before transformers to ensure all optimizations are applied correctly.
from unsloth import FastLanguageModel, PatchFastRL
from transformers.trainer_callback import TrainerCallback
from transformers import TrainingArguments, TrainerState, TrainerControl

random.seed(10042)
logger.add("grpo.log")

SUPPORTS_BFLOAT16 = False
if torch.cuda.is_available():
    major_version, minor_version = torch.cuda.get_device_capability()
    if major_version >= 8:
        SUPPORTS_BFLOAT16 = True
def is_bfloat16_supported():
    return SUPPORTS_BFLOAT16

# -------------------- Model --------------------
def load_model_and_tokenizer(model_path: str, max_seq_length: int = 8192, lora_rank: int = 64):
    PatchFastRL("GRPO", FastLanguageModel)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = max_seq_length,
        load_in_4bit = True, # False for LoRA 16bit
        fast_inference = True, # Enable vLLM fast inference
        max_lora_rank = lora_rank,
        gpu_memory_utilization = 0.85, # Reduce if out of memory
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ], # Remove QKVO if out of memory
        lora_alpha = lora_rank,
        use_gradient_checkpointing = "unsloth", # Enable long context finetuning
        random_state = 10042,
    )

    return model, tokenizer

# -------------------- Dataset --------------------

from task_rewards import reward_funcs

import sys, torch
import gc, ctypes
def clean_memory():
    gc.collect()
    if sys.platform == 'linux':
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # mps backend
    if torch.backends.mps.is_available():
        torch.backends.mps.rc.reset()

def build_trainer_callbacks(model, tokenizer, save_steps=10):
    class CacheFlushCallback(TrainerCallback):  # Inherit from a base Callback class
        def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
            if state.global_step % 1 == 0: # Flush cache every 10 steps
                clean_memory() # Clean memory
                print("Cache flushed at step:", state.global_step) # Optional: for monitoring
    cache_flush_callback = CacheFlushCallback()

    def save_model(model_weights_dir, model, tokenizer):
        model_weights_dir = "model_weights"
        logger.info(f"Saving model to {model_weights_dir}")
        model.save_pretrained_merged(model_weights_dir, tokenizer, save_method = "merged_16bit")
        logger.info(f"Model saved to {model_weights_dir}")

        adapter_model_dir = f"{model_weights_dir}/adapter_model"
        logger.info(f"Saving LoRA adapter model to {adapter_model_dir}")
        model.save_lora(adapter_model_dir)
        logger.info(f"LoRA adapter model saved to {adapter_model_dir}")

    class SaveModelCallback(TrainerCallback):  # Inherit from a base Callback class
        def __init__(self, save_steps: int = 0):
            self.save_steps = save_steps

        def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
            if self.save_steps > 0 and state.global_step % self.save_steps == 0:
                model_weights_dir = f"./output_grpo/ckpt/iter-{state.global_step:05d}"
                save_model(model_weights_dir=model_weights_dir, model=model, tokenizer=tokenizer)

        def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
            if self.save_steps == 0:
                model_weights_dir = f"./output_grpo/ckpt/iter-{state.global_step:05d}"
                save_model(model_weights_dir=model_weights_dir, model=model, tokenizer=tokenizer)

    save_model_callback = SaveModelCallback(save_steps=save_steps)

    callbacks = [cache_flush_callback, save_model_callback]
    return callbacks

# -------------------- Train --------------------
def train():

    from task_dataset import load_task_datasets
    dataset_path = "/opt/local/datasets/llm-reasoning/natural_reasoning_finance"
    model_path="/opt/local/llm_models/huggingface.co/Qwen/Qwen3-4B"

    train_dataset, eval_dataset = load_task_datasets(dataset_path)
    model, tokenizer = load_model_and_tokenizer(model_path, max_seq_length=4096, lora_rank=32)

    eval_steps=10
    save_steps=10

    from trl import GRPOConfig, GRPOTrainer
    training_args = GRPOConfig(
        use_vllm = True, # use vLLM for fast inference!
        learning_rate = 5e-6,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type = "cosine",
        optim = "adamw_8bit",
        logging_steps = 1,
        bf16 = is_bfloat16_supported(),
        fp16 = not is_bfloat16_supported(),
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4, # Increase to 4 for smoother training
        temperature=1.0,
        min_p=0.2,
        num_generations = 4, # Decrease if out of memory
        max_prompt_length = 2048,
        max_completion_length = 2048,
        num_train_epochs = 1, # Set to 1 for a full training run
        # max_steps = 250,
        do_eval = True,
        eval_steps = eval_steps,
        save_steps = save_steps,
        # save_strategy = "epoch",
        max_grad_norm = 0.1,
        report_to = "tensorboard", # Can use Weights & Biases, tensorboard, none
        output_dir = "outputs_grpo",
    )

    callbacks = build_trainer_callbacks(model, tokenizer, save_steps=save_steps)

    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = reward_funcs,
        args = training_args,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        save_total_limit = 1,
        callbacks=callbacks
    )

    trainer.train()
    # import torch.distributed as dist
    # try:
    #     trainer.train()
    # except Exception as e:
    #     logger.error(f"Training failed: {e}")
    # finally:
    #     dist.destroy_process_group()

def inference():
    # from unsloth_inference import model_inference
    # gen_kwargs = {
    #     "temperature": 0.6,
    #     "max_tokens": 8192,
    # }

    # prompt = "Which is bigger, 9.9 or 9.11?"
    # generated_text = model_inference(prompt, model=model, tokenizer=tokenizer, gen_kwargs=gen_kwargs, lora_request=adapter_model_dir)
    # logger.debug(f"{generated_text=}")
    pass



# # -------------------- Save Model --------------------

# # Merge to 16bit
# if True: model.save_pretrained_merged("model_fp16", tokenizer, save_method = "merged_16bit",)
# if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_16bit", token = "")

# # Merge to 4bit
# if False: model.save_pretrained_merged("model_4bit", tokenizer, save_method = "merged_4bit_forced",)
# if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_4bit", token = "")

# # Just LoRA adapters
# if True: model.save_pretrained_merged("model_lora", tokenizer, save_method = "lora",)
# if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "lora", token = "")


# # -------------------- Save GGUF --------------------

# # Save to 8bit Q8_0
# if False: model.save_pretrained_gguf("model_q8_0_gguf", tokenizer,)
# # Remember to go to https://huggingface.co/settings/tokens for a token!
# # And change hf to your username!
# if False: model.push_to_hub_gguf("hf/model", tokenizer, token = "")

# # Save to 16bit GGUF
# if False: model.save_pretrained_gguf("model", tokenizer, quantization_method = "f16")
# if False: model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "f16", token = "")

# # Save to q4_k_m GGUF
# if False: model.save_pretrained_gguf("model_q4_k_m_gguf", tokenizer, quantization_method = "q4_k_m")
# if False: model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "q4_k_m", token = "")

# # Save to multiple GGUF options - much faster if you want multiple!
# if False:
#     model.push_to_hub_gguf(
#         "hf/model", # Change hf to your username!
#         tokenizer,
#         quantization_method = ["q4_k_m", "q8_0", "q5_k_m",],
#         token = "",
#     )


def main():
    train()
    
if __name__ == '__main__':
    main()