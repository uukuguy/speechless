#!/usr/bin/env python
import os, json
import random
import torch
import numpy as np
from copy import deepcopy
from loguru import logger
from tqdm import tqdm
from typing import Any
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


import gc, ctypes
def clean_memory():
    for _ in range(3):
        gc.collect()
        if sys.platform == 'linux':
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        # mps backend
        if torch.backends.mps.is_available():
            torch.cuda.empty_cache()


# -------------------- Model --------------------
def load_model_and_tokenizer(model_path: str, max_seq_length: int = 8192, lora_rank: int = 64):
    from unsloth import FastLanguageModel, PatchFastRL
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


    model.config.use_cache = False # Disables KV caching to save memory.
    # Then enable gradient checkpointing
    model.gradient_checkpointing_enable()


    return model, tokenizer


# -------------------- Dataset --------------------

import re
from datasets import load_dataset, Dataset

# # Load and prep dataset
# SYSTEM_PROMPT = """
# Respond in the following format:
# <reasoning>
# ...
# </reasoning>
# <answer>
# ...
# </answer>
# """

# XML_COT_FORMAT = """\
# <reasoning>
# {reasoning}
# </reasoning>
# <answer>
# {answer}
# </answer>
# """
# def extract_xml_answer(text: str) -> str:
#     answer = text.split("<answer>")[-1]
#     answer = answer.split("</answer>")[0]
#     return answer.strip()

# def extract_hash_answer(text: str) -> str | None:
#     if "####" not in text:
#         return None
#     return text.split("####")[1].strip()

# # uncomment middle messages for 1-shot prompting
# def get_gsm8k_questions(split = "train") -> Dataset:
#     data = load_dataset('openai/gsm8k', 'main')[split] # type: ignore
#     data = data.map(lambda x: { # type: ignore
#         'prompt': [
#             {'role': 'system', 'content': SYSTEM_PROMPT},
#             {'role': 'user', 'content': x['question']}
#         ],
#         'answer': extract_hash_answer(x['answer'])
#     }) # type: ignore
#     return data # type: ignore

# dataset = get_gsm8k_questions()

from speechless.finetune.dataset_utils.multi_rounds import format_chat
def get_function_calling_dialogs(dataset_path):
    ds = load_dataset("json", data_files=dataset_path, split="train")

    def filter_func(example):
        messages_str = example['messages']
        # logger.debug(f"{messages_str=}")
        messages = json.loads(messages_str)
        if (len(messages) - 1) % 2 != 0:
            return False
        
        if messages[-1]['role'] != 'assistant':
            return False

        return True

    ds = ds.filter(filter_func)

    def add_targets_func(example):
        messages_str = example['messages']
        # logger.debug(f"{messages_str=}")
        messages = json.loads(messages_str)
        assert messages[-1]['role'] == 'assistant', f"{messages=}"
        # if messages[-1]['role'] == 'user':
        #     messages = messages[:-1]

        # logger.debug(f"{messages=}")
        assert messages[0]['role'] == "system"
        targets = []
        assert (len(messages) - 1) % 2 == 0, f"(len(messages) - 1) % 2 != 0. {len(messages)=}, {messages=}"

        for i in range(0, (len(messages) - 1) // 2):
            conv = messages[1 + i*2 + 1]
            tool_calls = conv.get("tool_calls", None)
            if tool_calls:
                target = [tool_calls[0]['function']]
            else:
                target = []
            targets.append(target)
                
        messages = json.dumps(messages, ensure_ascii=False)
        targets = json.dumps(targets, ensure_ascii=False)
        # logger.info(f"{messages=}, {targets=}")
        return {
            "messages": messages,
            "targets": targets,
        }

    ds = ds.map(add_targets_func)
    print(f"{ds=}, {ds[0]=}")

    expanded_examples = []
    for example in tqdm(ds, desc="Building examples"):
        messages = json.loads(example['messages'])
        targets = json.loads(example['targets'])
        if isinstance(targets, str):
            targets = json_loads(targets)
        assert len(messages) == len(targets) * 2 + 1, f"{len(messages)=}, {len(targets)=}"
        system_message = deepcopy(messages[0])
        messages = messages[1:]

        sub_examples = []
        for i in range(0, len(targets)):
            sub_messages = deepcopy(messages[:i*2+1])
            sub_targets = deepcopy(targets[:i+1])
            sub_example = {
                "messages": [system_message] + sub_messages,
                "targets": sub_targets,
            }
            sub_examples.append(sub_example)

        assert len(sub_examples) == len(targets), f"{len(sub_examples)=}, {len(targets)=}"

        # example['messages'] = [system_message] + messages[:-1]
        # example['targets'] = targets
        # sub_examples.append(example)

        selected_examples = []
        if len(sub_examples) > 1:
            random.shuffle(sub_examples)
            for e in sub_examples:
                targets = e['targets']
                if isinstance(targets, str):
                    targets = json_loads(targets)
                # logger.debug(f"{targets=}")
                if len(targets[-1]) == 0:
                    logger.debug(f"{targets=}")
                    # logger.debug(f"{e=}")
                    # logger.debug(f"e['targets'][-1] == [] found")
                    e['messages'] = json.dumps(e['messages'], ensure_ascii=False)
                    e['targets']  = json.dumps(e['targets'], ensure_ascii=False)
                    selected_examples.append(e)
                    break
            assert len(selected_examples) == 1, f"{sub_examples=}"
            for e in sub_examples:
                targets = e['targets']
                if isinstance(targets, str):
                    targets = json_loads(targets)
                # logger.debug(f"{targets=}")
                if len(targets[-1]) > 0:
                    _targets=e['targets']
                    logger.debug(f"{targets=}")
                    # logger.debug(f"{e=}")
                    # logger.debug(f"e['targets'][-1] != [] found")
                    e['messages'] = json.dumps(e['messages'], ensure_ascii=False)
                    e['targets']  = json.dumps(e['targets'], ensure_ascii=False)
                    selected_examples.append(e)
                    break
            assert len(selected_examples) == 2, f"{sub_examples=}"
            # print(f"{len(sub_examples)}, {sub_examples=}")
            assert len(selected_examples) == 2, f"{len(selected_examples)}, {selected_examples=}"
        else:
            # e = sub_examples[0]
            # e['messages'] = json.dumps(e['messages'], ensure_ascii=False)
            # e['targets']  = json.dumps(e['targets'], ensure_ascii=False)
            # selected_examples.append(e)
            pass
        expanded_examples.extend(selected_examples)

        # if len(sub_examples) >= 1:
        #     
        #     sub_example = sub_examples[0]
        #     expanded_examples.append(sub_example)
        
        

    ds = Dataset.from_list(expanded_examples)
    print(f"{ds=}, {ds[0]=}")

    def format_chat_func(example):
        # print(f"{example=}")
        messages = json.loads(example['messages'])
        # print(f"{type(messages)=}, {messages=}")
        if isinstance(messages, str):
            messages = json.loads(messages)
        prompt = format_chat(messages)

        return {
            "prompt": prompt
        }
        

    ds = ds.map(format_chat_func)
    ds = ds.remove_columns(["messages"])
    return ds

def get_function_calling_dialogs_sft_dataset(dataset_path):
    ds = load_dataset("json", data_files=dataset_path, split="train")

    def filter_func(example):
        messages_str = example['messages']
        # logger.debug(f"{messages_str=}")
        messages = json.loads(messages_str)
        if (len(messages) - 1) % 2 != 0:
            return False
        
        if messages[-1]['role'] != 'assistant':
            return False

        return True

    ds = ds.filter(filter_func)

    def add_targets_func(example):
        messages_str = example['messages']
        # logger.debug(f"{messages_str=}")
        messages = json.loads(messages_str)
        assert messages[-1]['role'] == 'assistant', f"{messages=}"
        # if messages[-1]['role'] == 'user':
        #     messages = messages[:-1]

        # logger.debug(f"{messages=}")
        assert messages[0]['role'] == "system"
        targets = []
        assert (len(messages) - 1) % 2 == 0, f"(len(messages) - 1) % 2 != 0. {len(messages)=}, {messages=}"

        for i in range(0, (len(messages) - 1) // 2):
            conv = messages[1 + i*2 + 1]
            tool_calls = conv.get("tool_calls", None)
            if tool_calls:
                target = [tool_calls[0]['function']]
            else:
                target = []
            targets.append(target)
                
        messages = json.dumps(messages, ensure_ascii=False)
        targets = json.dumps(targets, ensure_ascii=False)
        # logger.info(f"{messages=}, {targets=}")
        return {
            "messages": messages,
            "targets": targets,
        }

    ds = ds.map(add_targets_func)
    print(f"{ds=}, {ds[0]=}")

    expanded_examples = []
    for example in tqdm(ds, desc="Building examples"):
        messages = json.loads(example['messages'])
        targets = json.loads(example['targets'])
        if isinstance(targets, str):
            targets = json_loads(targets)
        assert len(messages) == len(targets) * 2 + 1, f"{len(messages)=}, {len(targets)=}"
        system_message = deepcopy(messages[0])
        messages = messages[1:]

        sub_examples = []
        for i in range(0, len(targets)):
            sub_messages = deepcopy(messages[:i*2+1])
            sub_targets = deepcopy(targets[:i+1])
            sub_example = {
                "messages": [system_message] + sub_messages,
                "targets": sub_targets,
            }
            sub_examples.append(sub_example)

        assert len(sub_examples) == len(targets), f"{len(sub_examples)=}, {len(targets)=}"

        # example['messages'] = [system_message] + messages[:-1]
        # example['targets'] = targets
        # sub_examples.append(example)

        selected_examples = []
        if len(sub_examples) > 1:
            random.shuffle(sub_examples)
            for e in sub_examples:
                targets = e['targets']
                if isinstance(targets, str):
                    targets = json_loads(targets)
                # logger.debug(f"{targets=}")
                if len(targets[-1]) == 0:
                    logger.debug(f"{targets=}")
                    # logger.debug(f"{e=}")
                    # logger.debug(f"e['targets'][-1] == [] found")
                    e['messages'] = json.dumps(e['messages'], ensure_ascii=False)
                    e['targets']  = json.dumps(e['targets'], ensure_ascii=False)
                    selected_examples.append(e)
                    break
            assert len(selected_examples) == 1, f"{sub_examples=}"
            for e in sub_examples:
                targets = e['targets']
                if isinstance(targets, str):
                    targets = json_loads(targets)
                # logger.debug(f"{targets=}")
                if len(targets[-1]) > 0:
                    _targets=e['targets']
                    logger.debug(f"{targets=}")
                    # logger.debug(f"{e=}")
                    # logger.debug(f"e['targets'][-1] != [] found")
                    e['messages'] = json.dumps(e['messages'], ensure_ascii=False)
                    e['targets']  = json.dumps(e['targets'], ensure_ascii=False)
                    selected_examples.append(e)
                    break
            assert len(selected_examples) == 2, f"{sub_examples=}"
            # print(f"{len(sub_examples)}, {sub_examples=}")
            assert len(selected_examples) == 2, f"{len(selected_examples)}, {selected_examples=}"
        else:
            # e = sub_examples[0]
            # e['messages'] = json.dumps(e['messages'], ensure_ascii=False)
            # e['targets']  = json.dumps(e['targets'], ensure_ascii=False)
            # selected_examples.append(e)
            pass
        expanded_examples.extend(selected_examples)

        # if len(sub_examples) >= 1:
        #     
        #     sub_example = sub_examples[0]
        #     expanded_examples.append(sub_example)
        
        

    ds = Dataset.from_list(expanded_examples)
    print(f"{ds=}, {ds[0]=}")

    def format_chat_func(example):
        # print(f"{example=}")
        messages = json.loads(example['messages'])
        # print(f"{type(messages)=}, {messages=}")
        if isinstance(messages, str):
            messages = json.loads(messages)
        prompt = format_chat(messages)

        return {
            "prompt": prompt
        }
        

    ds = ds.map(format_chat_func)
    ds = ds.remove_columns(["messages"])
    return ds


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

# def format_reward_func(prompts, completions, targets, **kwargs):
#     """
#     Format: <think>...</think><answer>...</answer>
#     Args:
#         completions (list[str]): Generated outputs
#         target (list[str]): Expected answers
      
#       Returns:
#           list[float]: Reward scores
#     """
#     rewards = []

#     for completion, gt in zip(completions, targets):

#         try:
#             # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
#             completion = "<think>" + completion
#             # if random.random() < 0.1:  # 1% chance to write samples into a file
#             #     os.makedirs("completion_samples", exist_ok=True)
#             #     log_file = os.path.join("completion_samples", "completion_samples.txt")
#             #     with open(log_file, "a") as f:
#             #         f.write(f"\n\n==============\n")
#             #         f.write(completion)
#             score = 0.0
#             if "<think>" in completion: score += 0.5
#             if "</think>" in completion: score += 0.5
#             # if "<answer>" in completion: score += 0.25
#             # if "</answer>" in completion: score += 0.25

#             # Check if the format is correct
#             # regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"

#             # match = re.search(regex, completion, re.DOTALL)
#             # # if the format is not correct, reward is 0
#             # if match is None or len(match.groups()) != 2:
#             #     score += 0.0
#             #     # rewards.append(0.0)
#             #     # logger.warning(f"严格匹配<think><answer>, 奖励0.0")
#             # else:
#             #     score += 1.0
#             #     # rewards.append(1.0)
#                 # logger.info(f"严格匹配<think><answer>, 奖励1.0")

#             if score < 1: 
#                 logger.warning(f"格式匹配<think>奖励: {score:.2f}")
#             else:
#                 logger.info(f"格式匹配<think>奖励: {score:.2f}")

#             rewards.append(score)
#         except Exception:
#             rewards.append(0.0)

#     logger.info(f"{rewards=}")
#     return rewards

# def correctness_reward_func(prompts, completions, targets, **kwargs) -> list[float]:
#     # responses = [completion[0]['content'] for completion in completions]
#     responses = [completion.strip() for completion in completions]

#     scores = []
#     for prompt, generated_text, target in zip(prompts, responses, targets):
#         logger.debug(f"{prompt=}")
#         logger.debug(f"generated_text=<think>{generated_text}")
#         score = 0.0
#         true_target = json_loads(target)
#         logger.info(f"{true_target=}")
#         if true_target[-1] == []:
#             if "<tool_call>" in generated_text and "</tool_call>" in generated_text: 
#                 # 在不应调用api的轮次，调用api，重点惩罚
#                 score = -1
#                 logger.warning(f"在不应调用api的轮次，调用api，重点惩罚！{score:.2f}")
#             else:
#                 # 在不应调用api的轮次，没有调用api，正常奖励
#                 score = 1
#                 logger.info(f"在不应调用api的轮次，没有调用api，正常奖励. {score:.2f}")
#         else:
#             if "<tool_call>" in generated_text and "</tool_call>" in generated_text: 
#                 score = 0.0
#                 # 在应该调用api的轮次，触发调用api，奖励
#                 logger.info(f"在应该调用api的轮次，触发调用api，奖励")
#                 # if generated_text[:len("<tool_call>")] == "<tool_call>" and generated_text[-len("</tool_call>")] == "</tool_call>":
#                 tool_call_text = generated_text.split("<tool_call>")[1].split("</tool_call>")[0]
#                 bad_tool_call_text = False

#                 try:
#                     # func = json.loads(tool_call_text)
#                     func = json_loads(tool_call_text)
#                 except Exception as e:
#                     # api 不是正确的json格式，虽然触发时机正确，但不得分 
#                     bad_tool_call_text = True
#                     score = -2.0
#                     logger.error(f"<tool_call> api json 解释失败，重点惩罚！{score:.2f}")


#                 if not bad_tool_call_text:
#                     if len(re.findall(r"<tool_call>", generated_text)) == 1 and len(re.findall(r"<tool_call>", generated_text)) == 1:
#                         logger.info(f"<tool_call>对只能出现一次，符合限制条件，奖励0.5")
#                         score += 0.5
#                     else:
#                         score += -0.5
#                         logger.warning(f"<tool_call>对只能出现一次，不符合限制条件，重点惩罚！{score:.2f}")
#                         bad_tool_call_text = True

#                 if not bad_tool_call_text:
#                     if "name" in func and "arguments" in func:
#                         score += 0.5 # api 的json格式正确，格式奖励
#                         logger.info(f"api 的json格式keys('name, 'arguments)正确，格式奖励0.5")
#                     else:
#                         score += -0.5
#                         logger.warning(f"api 的json格式keys('name, 'arguments)错误，重点惩罚！{score:.2f}")
#                         bad_tool_call_text = True


#                 if not bad_tool_call_text:
#                     func_name = func['name']
#                     func_arguments = func['arguments']
#                     true_name = true_target[-1][0]['name']
#                     true_arguments = true_target[-1][0]['arguments']
#                     if func_name == true_name:
#                         score += 1.0 # 函数名正确，
#                         logger.info(f"函数名{func_name}正确，奖励1.0")

#                         func_argument_keys = set(func_arguments.keys())
#                         true_argument_keys = set(true_arguments.keys())
#                         if func_argument_keys == true_argument_keys:
#                             score += 1.0 # 参数名完全一致，奖励
#                             logger.info(f"参数名完全一致，奖励1.0")
#                         else:
#                             tp = len(func_argument_keys & true_argument_keys)
#                             fp = len(func_argument_keys - true_argument_keys)
#                             fn = len(true_argument_keys - func_argument_keys)
#                             if (tp+fp) > 0 and (tp+fn) > 0:
#                                 p = tp / (tp+fp)
#                                 r = tp / (tp+fn)
#                                 if (p+r) > 0:
#                                     f1 = 2 * p * r / (p+r)
#                                 else:
#                                     f1 = 0.0
#                             else:
#                                 f1 = 0.0
#                             score += f1 # 参数名命中f1, 奖励    
#                             logger.info(f"参数名命中f1, 奖励{f1:.2f}")

#                         num_value_exist = 0
#                         for k, v in func_arguments.items():
#                             if f"{v}" in prompt:
#                                 num_value_exist += 1
#                         value_exist_acc = 0.0
#                         if len(func_arguments) > 0:
#                             value_exist_acc = num_value_exist / len(func_arguments)
#                             score += value_exist_acc
#                             logger.info(f"参数值在用户对话内容中命中率, 奖励{value_exist_acc:.2f}")

#                         num_not_null_values = 0
#                         num_str_values = 0
#                         for k, v in func_arguments.items():
#                             if isinstance(v, str):
#                                 num_str_values += 1
#                                 if v:
#                                     num_not_null_values += 1
#                         null_str_score = 0.0
#                         if num_str_values > 0 and num_not_null_values < num_str_values:
#                             null_str_score = -(num_str_values - num_not_null_values) / num_str_values
#                             score += null_str_score
#                             logger.warning(f"空串惩罚：{null_str_score:.2f}")

#                         same_keys = func_argument_keys & true_argument_keys
#                         tp = 0
#                         for k in same_keys:
#                             func_value = func_arguments[k]
#                             true_value = true_arguments[k]
#                             if f"{func_value}" == f"{true_value}":
#                                 tp += 1
#                         if len(func_argument_keys) > 0 and len(true_argument_keys) > 0:
#                             p = tp / len(func_argument_keys)
#                             r = tp / len(true_argument_keys)

#                             if (p+r) > 0:
#                                 f1 = 2 * p * r / (p+r)
#                             else:
#                                 f1 = 0.0
#                             if f1 > 0:
#                                 logger.info(f"参数值完全相同f1, 奖励{f1:.2f}")
#                             else:
#                                 logger.info(f"没有参数值完全相同, 惩罚-0.5")
#                                 f1 = -0.5
#                         else:
#                             f1 = 0.0
#                         score += f1 

#             else:
#                 # 在应该调用api的轮次没有调用api，重点惩罚
#                 score = -1
#                 logger.warning(f"在应该调用api的轮次没有调用api，重点惩罚！ {score:.2f}")


#         logger.debug(f"{score=}")
#         scores.append(score)

#     # format_scores = format_reward_func(prompts, completions, targets, **kwargs)
#     # scores = [ s0 + s1 for s0, s1 in zip(scores, format_scores)]

#     # logger.debug(f"{responses=}")
#     logger.info(f"{scores=}")
#     return scores


# -------------------- Reward Functions --------------------


def format_reward_func(prompts, completions, targets, **kwargs):
    """
    Format: <think>...</think><answer>...</answer>
    Args:
        completions (list[str]): Generated outputs
        target (list[str]): Expected answers
      
      Returns:
          list[float]: Reward scores
    """
    rewards = []

    for completion, gt in zip(completions, targets):

        try:
            # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
            completion = "<think>" + completion
            # if random.random() < 0.1:  # 1% chance to write samples into a file
            #     os.makedirs("completion_samples", exist_ok=True)
            #     log_file = os.path.join("completion_samples", "completion_samples.txt")
            #     with open(log_file, "a") as f:
            #         f.write(f"\n\n==============\n")
            #         f.write(completion)
            score = 0.0
            if "<think>" in completion: score += 0.25
            if "</think>" in completion: score += 0.25
            # if "<answer>" in completion: score += 0.25
            # if "</answer>" in completion: score += 0.25

            if len(re.findall(r"<think>", completion)) == 1 and len(re.findall(r"</think>", completion)) == 1:
                score += 0.5

            # Check if the format is correct
            # regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"

            # match = re.search(regex, completion, re.DOTALL)
            # # if the format is not correct, reward is 0
            # if match is None or len(match.groups()) != 2:
            #     score += 0.0
            #     # rewards.append(0.0)
            #     # logger.warning(f"严格匹配<think><answer>, 奖励0.0")
            # else:
            #     score += 1.0
            #     # rewards.append(1.0)
            # logger.info(f"严格匹配<think><answer>, 奖励1.0")

            if score < 1:
                logger.warning(f"格式匹配<think>奖励: {score:.2f}")
            else:
                logger.info(f"格式匹配<think>奖励: {score:.2f}")

            score = round(score, 3)
            rewards.append(score)
        except Exception:
            rewards.append(0.0)

    avg_reward = np.mean(rewards)
    if avg_reward < 1:
        logger.warning(f"格式匹配<think>奖励: {round(avg_reward, 3)}, {rewards}")
    else:
        logger.info(f"格式匹配<think>奖励: {round(avg_reward, 3)}, {rewards}")
    return rewards


def correctness_reward_func(prompts, completions, targets, **kwargs) -> list[float]:
    # responses = [completion[0]['content'] for completion in completions]
    responses = [completion.strip() for completion in completions]

    scores = []
    for prompt, generated_text, target in zip(prompts, responses, targets):
        # logger.debug(f"{prompt=}")
        # if "</think>" in generated_text:
        #     generated_text = "<think>" + generated_text

        generated_text = "<think>" + generated_text
        logger.debug(f"{generated_text=}")
        if not (len(re.findall(r"<think>", generated_text)) == 1 and len(re.findall(r"</think>", generated_text)) == 1):
            score = 0 
            logger.warning(f"<think>标签对只能出现一次，不符合限制条件，不再计算API调用细项得分！{score:.2f}")
            scores.append(score)
            continue

        score = 0.0
        true_target = json_loads(target)
        logger.info(f"{true_target=}")
        if true_target[-1] == []:
            if "<tool_call>" in generated_text and "</tool_call>" in generated_text:
                # 在不应调用api的轮次，调用api，重点惩罚
                score = 0.0
                logger.debug(f"在不应调用api的轮次，调用api，{score:.2f}")
            else:
                # 在不应调用api的轮次，没有调用api，正常奖励
                score = 1.0
                logger.info(f"在不应调用api的轮次，没有调用api，正常奖励. {score:.2f}")
        else:
            if "<tool_call>" in generated_text and "</tool_call>" in generated_text:
                score = 0.0
                # 在应该调用api的轮次，触发调用api，奖励
                logger.info(f"在应该调用api的轮次，触发调用api，奖励")
                # if generated_text[:len("<tool_call>")] == "<tool_call>" and generated_text[-len("</tool_call>")] == "</tool_call>":
                tool_call_text = generated_text.split("<tool_call>")[1].split("</tool_call>")[0]
                bad_tool_call_text = False

                try:
                    # func = json.loads(tool_call_text)
                    func = json_loads(tool_call_text)
                except Exception as e:
                    # api 不是正确的json格式，虽然触发时机正确，但不得分
                    bad_tool_call_text = True
                    score = -1
                    logger.error(f"<tool_call> api json 解释失败，{score:.2f}")

                # ---------- API 调用细项得分 ----------
                # ----- <tool_call>对只能出现一次，0.5
                if not bad_tool_call_text:
                    if len(re.findall(r"<tool_call>", generated_text)
                           ) == 1 and len(re.findall(r"<tool_call>", generated_text)) == 1:
                        # logger.info(f"<tool_call>对只能出现一次，符合限制条件，奖励0.5")
                        # score += 0.5
                        pass
                    else:
                        score = -1
                        logger.warning(f"<tool_call>对只能出现一次，不符合限制条件。{score:.2f}")
                        bad_tool_call_text = True

                # ----- api 的json格式keys('name, 'arguments)正确，0.5
                if not bad_tool_call_text:
                    ok = False
                    if "name" in func and "arguments" in func:
                        func_name = func["name"]
                        func_arguments = func["arguments"]
                        if func_name is not None and func_arguments is not None:
                            if isinstance(func_name, str) and isinstance(func_arguments, dict):
                                score += 0.5  # api 的json格式正确，格式奖励
                                logger.info(f"api 的json格式keys('name, 'arguments)正确，格式奖励0.5")
                                ok = True
                    if not ok:
                        score = -1
                        logger.warning(f"api 的json格式keys('name, 'arguments)错误。{score:.2f}")
                        bad_tool_call_text = True

                if not bad_tool_call_text:
                    score = 0.0
                    func_name = func['name']
                    func_arguments = func['arguments']
                    true_name = true_target[-1][0]['name']
                    true_arguments = true_target[-1][0]['arguments']
                    ## ----- 函数名正确，1.0
                    if func_name == true_name:
                        score += 0.25  # 函数名正确，
                        logger.info(f"函数名{func_name}正确，奖励0.25")

                        # ----- 参数名完全一致，1.0
                        func_argument_keys = set(func_arguments.keys())
                        true_argument_keys = set(true_arguments.keys())
                        if func_argument_keys == true_argument_keys:
                            score += 0.25  # 参数名完全一致，奖励
                            logger.info(f"参数名完全一致，奖励0.25")
                        else:
                            tp = len(func_argument_keys & true_argument_keys)
                            fp = len(func_argument_keys - true_argument_keys)
                            fn = len(true_argument_keys - func_argument_keys)
                            if (tp + fp) > 0 and (tp + fn) > 0:
                                p = tp / (tp + fp)
                                r = tp / (tp + fn)
                                if (p + r) > 0:
                                    f1 = 2 * p * r / (p + r)
                                else:
                                    f1 = 0.0
                            else:
                                f1 = 0.0
                            f1 *= 0.25
                            score += f1  # 参数名命中f1, 奖励
                            logger.info(f"参数名命中f1, 奖励{f1:.2f}")

                        # ----- 参数值在用户对话内容中命中率, 1.0
                        num_value_exist = 0
                        for k, v in func_arguments.items():
                            if f"{v}" in prompt:
                                num_value_exist += 1
                        value_exist_acc = 0.0
                        if len(func_arguments) > 0:
                            value_exist_acc = num_value_exist / len(func_arguments) * 0.25
                            score += value_exist_acc
                            logger.info(f"参数值在用户对话内容中命中率, 奖励{value_exist_acc:.2f}")

                        # ----- 空串惩罚，-1.0
                        num_not_null_values = 0
                        num_str_values = 0
                        for k, v in func_arguments.items():
                            if isinstance(v, str):
                                num_str_values += 1
                                if v:
                                    num_not_null_values += 1
                        null_str_score = 0.0
                        if num_str_values > 0 and num_not_null_values < num_str_values:
                            null_str_score = -(num_str_values - num_not_null_values) / num_str_values *.25
                            score += null_str_score
                            logger.warning(f"空串惩罚：{null_str_score:.2f}")

                        # ----- 参数值完全相同f1, 1.0
                        same_keys = func_argument_keys & true_argument_keys
                        tp = 0
                        for k in same_keys:
                            func_value = func_arguments[k]
                            true_value = true_arguments[k]
                            if f"{func_value}" == f"{true_value}":
                                tp += 1
                        if len(func_argument_keys) > 0 and len(true_argument_keys) > 0:
                            p = tp / len(func_argument_keys)
                            r = tp / len(true_argument_keys)

                            if (p + r) > 0:
                                f1 = 2 * p * r / (p + r)
                            else:
                                f1 = 0.0
                            if f1 > 0:
                                f1 *= 0.25
                                logger.info(f"参数值完全相同f1, 奖励{f1:.2f}")
                            else:
                                logger.info(f"没有参数值完全相同, 惩罚0.0")
                                f1 = 0
                        else:
                            f1 = 0.0
                        score += f1 

            else:
                # 在应该调用api的轮次没有调用api，重点惩罚
                score = -1.0
                logger.warning(f"在应该调用api的轮次没有调用api。 {score:.2f}")

        if score > 0:
            score = score * 2 
        logger.debug(f"score: {round(score, 3)}")
        scores.append(score)

    # format_scores = format_reward_func(prompts, completions, targets, **kwargs)
    # scores = [ s0 + s1 for s0, s1 in zip(scores, format_scores)]

    # logger.debug(f"{responses=}")
    arg_score = np.mean(scores)
    logger.info(f"API 调用正确性奖励: {round(arg_score, 3)}, {[round(s, 3) for s in scores]}")
    return scores


reward_funcs = [
    correctness_reward_func,
    format_reward_func,
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


# -------------------- Train --------------------
def train():
    dataset_path = "./data/rft_train_data_v6_0228_2.jsonl"
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    print(f"{dataset=}") 

# Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags.
# Think step by step inside <think> tags.
# - 仔细检查 API Schema 中函数的参数定义，当必填参数没有全部被用户明确提到时，明确告知用户缺少哪些必填参数，此轮次不要生成 api 调用。
    def r1_func(example):
        prompt = example['prompt']
        r1_prefix = """You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.

Show your work in <think> </think> tags. 
Think step by step inside <think> tags.

- 用户对话列表user_messages是用户的多轮对话内容，你需要判断在哪个轮次需要调用工具函数。整个对话轮次最多只有一个api调用，通常在用户关于工具函数参数的描述结束的那一轮。
- 你需要通过用户对话列表的内容准确判断工具函数的函数名，并从对话列表内容中抽取函数的参数值，同时参考 API Schema 中函数参数的说明，保证参数值的格式正确。
- 参数值尽量直接从对话内容中抽取，同时要重点关注日期、数字的正确格式。
- 不要对任何参数使用假设值、猜测值。
- 值为空的参数不要放入 api 调用中。
- 仔细检查 API Schema 中函数的参数定义，当必填参数没有全部被用户明确提到时，明确告知用户缺少哪些必填参数，此轮次用所有已知参数生成 api 调用。
- 请使用中文回复。

        """
        prompt = prompt.replace("<|im_start|>system\n\n",  "")
        prompt = "<|im_start|>system\n\n" + r1_prefix + prompt
        prompt += "<|im_start|>assistant\nLet me solve this step by step.\n<think>"

        targets = example['targets']
        return {
            "prompt": prompt,
            "targets": targets,
        }

    dataset = dataset.map(r1_func)

    eval_size = 200
    dataset = dataset.train_test_split(test_size=eval_size)
    train_dataset = dataset['train']
    eval_dataset = dataset['test']
    print(f"{train_dataset=}") 
    print(f"{eval_dataset=}") 

    # 20250302
    # model_path="/opt/local/llm_models/huggingface.co/speechlessai/function_calling_qwen_7b_instruct"
    # 20250303
    # model_path="/opt/local/llm_models/huggingface.co/speechlessai/function_calling_qwen_7b_instruct-unsloth"
    # 20250309
    model_path="/opt/local/llm_models/huggingface.co/speechlessai/function_calling_qwen_7b_instruct"
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
            model_weights_dir = f"./outputs_grpo_think/ckpt/iter-{state.global_step:05d}"
            save_model(model_weights_dir=model_weights_dir, model=model, tokenizer=tokenizer)
    save_model_callback = SaveModelCallback()

    from datetime import datetime
    run_name = "think-" + datetime.now().strftime("%Y%m%d_%H%M%S")
    from trl import GRPOConfig, GRPOTrainer
    training_args = GRPOConfig(
        use_vllm = True, # use vLLM for fast inference!
        learning_rate = 5e-6, #5e-7,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.03,
        lr_scheduler_type = "cosine",
        optim = "adamw_8bit",
        logging_steps = 1,
        bf16 = is_bfloat16_supported(),
        fp16 = not is_bfloat16_supported(),
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4, # Increase to 4 for smoother training
        num_generations = 4, # Decrease if out of memory
        max_prompt_length = 128,
        max_completion_length = 384,
        num_train_epochs = 1, # Set to 1 for a full training run
        # max_steps = 250,
        do_eval = True,
        eval_steps = 10,
        save_steps = 1000,
        # save_strategy = "epoch",
        max_grad_norm = 0.1,
        report_to = ["tensorboard", "wandb"], # Can use Weights & Biases, tensorboard, none
        run_name = run_name,
        output_dir = f"outputs_grpo_think/{run_name}",
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

def geneate_train_data():
    dataset_path = "/Users/sujiangwen/sandbox/LLM/speechless.ai/speechless/tasks/synthesize_tools_sft/data/function_calling_dialogs_v6_0228.jsonl"
    dataset = get_function_calling_dialogs(dataset_path)
    dataset.to_json("./rft_train_data_v6_0228_2.jsonl", force_ascii=False)
    

def main():
    train()
    # geneate_train_data()
    
if __name__ == '__main__':
    main()