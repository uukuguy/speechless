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

import re
from datasets import load_dataset, Dataset



# -------------------- Reward Functions --------------------
reward_funcs = []

def json_loads(json_str: str, ensure_ascii: bool = False, use_json_repair: bool = True) -> Any:
    if use_json_repair:
        from json_repair import repair_json
        return repair_json(json_str, return_objects=True, ensure_ascii=ensure_ascii)
    else:
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Error: {e}")
            return None

def correctness_reward_func(prompts, completions, targets, **kwargs) -> list[float]:
    # responses = [completion[0]['content'] for completion in completions]
    responses = [completion.strip() for completion in completions]

    scores = []
    for prompt, generated_text, true_target in zip(prompts, responses, targets):
        # logger.debug(f"{prompt[:30]=}")
        # logger.debug(f"{generated_text=}")
        score = 0.0
        # logger.info(f"{true_target=}")

        def get_boxed_value(text):
            boxed_value=None
            boxed_text = re.findall(r"boxed{(.*?)}", text, re.DOTALL | re.MULTILINE)
            if len(boxed_text) > 0:
                boxed_text = boxed_text[0]
                if boxed_text in ["1", "0"]:
                    boxed_value = int(boxed_text)
            return boxed_value

        true_value = get_boxed_value(true_target)
        llm_value = get_boxed_value(generated_text)

        if llm_value is not None: 
            if true_value == llm_value:
                score = 2.0
        if score == 0.0:
            logger.warning(f"{generated_text=}")
            logger.warning(f"{true_target=}")
            

        logger.debug(f"{score=}")
        scores.append(score)

    # logger.debug(f"{responses=}")
    logger.info(f"{scores=}")
    return scores

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    responses = [completion for completion in completions]
    pattern = r"<think>.*?</think>[\n\s]*\\boxed{[01]}"
    matches = [re.match(pattern, r, re.DOTALL | re.MULTILINE) for r in responses]
    scores = [0.5 if match else 0.0 for match in matches]

    pattern = r"^<think>\n.*?\n</think>\n\\boxed{[01]}"
    matches = [re.match(pattern, r, re.DOTALL | re.MULTILINE) for r in responses]
    for i, match in enumerate(matches):
        if match:
            scores[i] += 0.5
    return scores

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>[\n\s]*\\boxed{[01]}"
    responses = [completion for completion in completions]
    # matches = [re.match(pattern, r) for r in responses]
    # return [0.5 if match else 0.0 for match in matches]
    matches = [re.findall(pattern, r, re.DOTALL | re.MULTILINE) for r in responses]
    scores = [] 
    for m in matches:
        s = 0.0
        if len(m) >= 1:
            s += 0.5
            if len(m) == 1:
                s += 0.5
        scores.append(s)
    return scores

def think_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    responses = [completion for completion in completions]

    pattern = r"<think>.*?</think>"
    found = [0.5 if len(re.findall(pattern, r, re.DOTALL | re.MULTILINE)) == 1 else 0.0 for r in responses]
    return found

def boxed_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    responses = [completion for completion in completions]

    pattern = r"\\boxed{[01]}"
    found = [0.5 if len(re.findall(pattern, r, re.DOTALL | re.MULTILINE)) == 1 else 0.0 for r in responses]
    return found

def think_length_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion for completion in completions]
    thinkings = [ len(r.split("<think>")[-1].split("</think>")[0]) for r in responses]
    def get_score(length):
        min_length = 128
        max_length = 768
        if length < min_length or length > max_length:
            return 0.0
        else:
            return (length - min_length) / (max_length - min_length)
            
    return [get_score(t) for t in thinkings]

reward_funcs = [
    strict_format_reward_func,
    soft_format_reward_func,
    think_length_reward_func,
    think_format_reward_func,
    boxed_format_reward_func,
    correctness_reward_func,
]

# # Reward functions
# def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
#     responses = [completion[0]['content'] for completion in completions]
#     q = prompts[0][-1]['content']
#     extracted_responses = [extract_xml_answer(r) for r in responses]
#     logger.debug('-'*20 + 
#         f"\nQuestion:\n{q}" + 
#         f"\nAnswer:\n{answer[0]}" + 
#         f"\nResponse:\n{responses[0]}" + 
#         f"\nExtracted:\n{extracted_responses[0]}")
#     return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

# def int_reward_func(completions, **kwargs) -> list[float]:
#     responses = [completion[0]['content'] for completion in completions]
#     extracted_responses = [extract_xml_answer(r) for r in responses]
#     return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

# def strict_format_reward_func(completions, **kwargs) -> list[float]:
#     """Reward function that checks if the completion has a specific format."""
#     pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
#     responses = [completion[0]["content"] for completion in completions]
#     matches = [re.match(pattern, r) for r in responses]
#     return [0.5 if match else 0.0 for match in matches]

# def soft_format_reward_func(completions, **kwargs) -> list[float]:
#     """Reward function that checks if the completion has a specific format."""
#     pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
#     responses = [completion[0]["content"] for completion in completions]
#     matches = [re.match(pattern, r) for r in responses]
#     return [0.5 if match else 0.0 for match in matches]

# def count_xml(text) -> float:
#     count = 0.0
#     if text.count("<reasoning>\n") == 1:
#         count += 0.125
#     if text.count("\n</reasoning>\n") == 1:
#         count += 0.125
#     if text.count("\n<answer>\n") == 1:
#         count += 0.125
#         count -= len(text.split("\n</answer>\n")[-1])*0.001
#     if text.count("\n</answer>") == 1:
#         count += 0.125
#         count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
#     return count

# def xmlcount_reward_func(completions, **kwargs) -> list[float]:
#     contents = [completion[0]["content"] for completion in completions]
#     return [count_xml(c) for c in contents]

# reward_funcs = [
#     xmlcount_reward_func,
#     soft_format_reward_func,
#     strict_format_reward_func,
#     int_reward_func,
#     correctness_reward_func,
# ]


# dataset_path = "/Users/sujiangwen/sandbox/LLM/speechless.ai/speechless/tasks/synthesize_tools_sft/data/function_calling_dialogs_v6_0228.jsonl"
# dataset = get_function_calling_dialogs(dataset_path)
# dataset.to_json("./rft_train_data_v6_0228.jsonl", force_ascii=False)

# dataset_path = "./rft_train_data_v6_0228.jsonl"
# dataset = load_dataset("json", data_files=dataset_path, split="train")
# print(f"{dataset=}, {dataset[:10]=}")

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


from speechless.finetune.dataset_utils.multi_rounds import format_chat
# -------------------- Train --------------------
def train():

    from prompt_gemini import SYSTEM_PROMPT, USER_PROMPT

    dataset_path = "./reserved_data_model_1.jsonl"
    # dataset_path = "./train_data_model_1.jsonl"
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    print(f"{dataset=}") 

    def generate_prompt_func(example):
        text = example['text']
        label = example.get('label', None)

        system_prompt = SYSTEM_PROMPT
        user_prompt = USER_PROMPT.format(text=text[:1024])

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        prompt = format_chat(messages)
        if label is not None:
            response = "<think></think>\n\\boxed{" + str(label) + "}"
        else:
            response = ""

        return {
            "prompt": prompt,
            "targets": response,
        }

    dataset = dataset.map(generate_prompt_func)
    dataset = dataset.remove_columns(['text','label'])
    print(f"final dataset: {dataset}") 

    eval_size = 200
    dataset = dataset.train_test_split(test_size=eval_size)
    train_dataset = dataset['train']
    eval_dataset = dataset['test']
    print(f"{train_dataset=}") 
    print(f"{eval_dataset=}") 

    # 20250302
    # model_path="/opt/local/llm_models/huggingface.co/speechlessai/function_calling_qwen_7b_instruct"
    # 20250303
    # model_path="/opt/local/llm_models/huggingface.co/speechlessai/gen-text-detector-Qwen3-4B-gemini-1"

    # 20250523
    # model_path="/opt/local/llm_models/huggingface.co/Qwen/Qwen3-4B"
    # model_path="/opt/local/llm_models/huggingface.co/speechlessai/gen-text-detector-rft-Qwen3-4B-unsloth"

    # 20250526
    # model_path="/opt/local/llm_models/huggingface.co/speechlessai/gen-text-detector-Qwen3-4B-gemini-1"

    # model_path="/opt/local/llm_models/huggingface.co/speechlessai/gen-text-detector-Qwen3-4B-eval-correct-cot"

    # 20250529
    model_path="/opt/local/llm_models/huggingface.co/speechlessai/gen-text-detector-Qwen3-4B-gemini-1"


    model, tokenizer = load_model_and_tokenizer(model_path)

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
        def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
            model_weights_dir = f"./output_grpo/ckpt/iter-{state.global_step:05d}"
            save_model(model_weights_dir=model_weights_dir, model=model, tokenizer=tokenizer)
    save_model_callback = SaveModelCallback()


    from trl import GRPOConfig, GRPOTrainer
    training_args = GRPOConfig(
        use_vllm = True, # use vLLM for fast inference!
        learning_rate = 2e-5, #5e-6,
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
        max_prompt_length = 512,
        max_completion_length = 1024,
        num_train_epochs = 3, # Set to 1 for a full training run
        # max_steps = 250,
        do_eval = True,
        eval_steps = 10,
        save_steps = 500,
        # save_strategy = "epoch",
        max_grad_norm = 0.1,
        report_to = "tensorboard", # Can use Weights & Biases, tensorboard, none
        output_dir = "outputs_grpo",
    )


    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = reward_funcs,
        args = training_args,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        callbacks=[cache_flush_callback, save_model_callback]
    )

    trainer.train()
    # import torch.distributed as dist
    # try:
    #     trainer.train()
    # except Exception as e:
    #     logger.error(f"Training failed: {e}")
    # finally:
    #     dist.destroy_process_group()

    # saved_lora_dir = "grpo_saved_lora"
    # model.save_lora(saved_lora_dir)

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