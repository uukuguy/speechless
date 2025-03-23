#!/usr/bin/env python

import json, re, os, sys
from tqdm import tqdm
from loguru import logger

from enum import Enum
class Difficulty(Enum):
    EASY = 0 
    MEDIUM = 1 
    HARD = 2


def get_answer_targets_count(targets):
    return sum([t != [] for t in targets])


def targets_is_null(targets):
    return all([t == [] for t in targets])


def get_last_target(targets):
    # Reverse the order of targets to get the last target
    for target in reversed(targets):
        if target != []:
            return target
    return []

def check_value_type(value, value_type):
    if value_type == "string":
        return isinstance(value, str)
    elif value_type == "integer":
        return isinstance(value, int)
    elif value_type == "float":
        return isinstance(value, float)
    elif value_type == "number":
        return isinstance(value, int) or isinstance(value, float)
        # if not isinstance(value, int) and not isinstance(value, float):
        #     print(f"value: {value} ({type(value)}), value_type: {value_type}")
        #     logger.warning(f"check_value_type: {type(value)}, {value} is not a number")
        #     return False
    elif value_type == "boolean":
        return isinstance(value, bool)
    elif value_type == "array":
        return isinstance(value, list)
    elif value_type == "dict":
        return isinstance(value, dict)

    return False

def check_string_value(id, arg_value, arg_description, user_messages_content):
    if not check_value_type(arg_value, "string"):
        return False

    if "YYYY-MM-DD HH:MM:SS" in arg_description:
        if not re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", arg_value):
            return False
    elif "YYYY-MM-DDTHH:mm:ssZ" in arg_description:
        if not re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", arg_value):
            return False
    elif "YYYY-MM-DD" in arg_description:
        if not re.match(r"\d{4}-\d{2}-\d{2}", arg_value):
            return False
    elif "YYYY-MM-DD HH:MM" in arg_description:
        if not re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", arg_value):
            return False
    elif "YYYY-MM-DD HH:mm" in arg_description:
        if not re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", arg_value):
            return False
    elif "HH:MM" in arg_description:
        if not re.match(r"\d{2}:\d{2}", arg_value):
            return False
    elif "HH:mm" in arg_description:
        if not re.match(r"\d{2}:\d{2}", arg_value):
            return False
    elif "ISO 08601" in arg_description: 
        if not re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", arg_value):
            return False
    else:
        if "日期" not in arg_description and "时间" not in arg_description:
            if arg_value not in user_messages_content:
                # logger.debug(f"{id=}, {arg_value=}, {user_messages_content=}")
                return False
    return True

def check_number_value(id, arg_value, arg_description):
    if not check_value_type(arg_value, "number"):
        return False

    if "时间戳" in arg_description:
        if not re.match(r"\d{10}", str(arg_value)):
            return False
    return True

def check_boolean_value(id, arg_value):
    if not check_value_type(arg_value, "boolean"):
        return False
    return True

def check_value(id, value, value_type, value_description, user_messages_content):
    if value_type == "string":
        return check_string_value(id, value, value_description, user_messages_content)
    if value_type == "number":
        return check_number_value(id, value, value_description)
    if value_type == "boolean":
        return check_boolean_value(id, value)
    return True

def check_arg_value_in_user_messages(id, arg_value, arg_type, arg_description, user_messages_content) -> float:
    if arg_type in ["number", "integer", "float"]:
        if str(arg_value) in user_messages_content:
            return True
        else:
            # logger.debug(f"{id}, {arg_value=}, {user_messages_content=}")
            return False
    return True

def check_function_args(id, func_args, api_args, user_messages_content):
    # logger.info(user_messages_content)

    if len(func_args) == 0:
        return 0

    num_wrong = 0
    for arg_name, arg_value in func_args.items():
        api_arg = api_args[arg_name]
        arg_type = api_arg["type"]
        arg_description = api_arg["description"]

        if not check_value_type(arg_value, arg_type):
            num_wrong += 1
            # logger.debug(f"{arg_name=}, {arg_value=}, {arg_type=}, {arg_description=}")
            continue

        if not check_value(id, arg_value, arg_type, arg_description, user_messages_content):
            num_wrong += 1
            # logger.debug(f"{arg_name=}, {arg_value=}, {arg_type=}, {arg_description=}")
            continue

        if not check_arg_value_in_user_messages(id, arg_value, arg_type, arg_description, user_messages_content):
            num_wrong += 1
            # logger.debug(f"{arg_name=}, {arg_value=}, {arg_type=}, {arg_description=}")
            continue

        elif arg_type == "array":
            if "items" in api_arg:
                item_type = api_arg["items"]["type"]
                for v in arg_value:
                    if not check_value(id, v, item_type, arg_description, user_messages_content):
                        # logger.debug(f"{arg_name=}, {arg_value=}, {arg_type=}, {arg_description=}")
                        num_wrong += 1 
                        break

        elif arg_type == "dict":
            if not isinstance(arg_value, dict):
                return False
            for k, v in arg_value.items():
                if not check_value(id, v, arg_type, arg_description, user_messages_content):
                    num_wrong += 1 
                    # logger.debug(f"{arg_name=}, {arg_value=}, {arg_type=}, {arg_description=}")
                    break
    
    wrong_ratio = num_wrong / len(func_args) 
    # logger.warning(f"{num_wrong=}, {len(func_args)=}, {wrong_ratio=}")
    # if wrong_ratio >= 0.5:
    #     logger.warning(f"{func_args=}")
    #     logger.info(f"{api_args=}")

    return wrong_ratio


def check_function(id, function, apis, user_messages) -> Difficulty:
    func_name = function["name"]
    func_args = function["arguments"]

    apis_dict = {api["name"]: api for api in apis}
    api = apis_dict[func_name]
    api_args = api["parameters"]['properties']

    user_messages_content = "\n".join([m for m in user_messages])

    # 函数名不在API名中
    if func_name not in apis_dict:
        return Difficulty.HARD

    # 函数参数名不在API参数名中
    for arg_name in func_args.keys():
        if arg_name not in api_args:
            return Difficulty.HARD

    # 必填参数不全
    required = api["parameters"]["required"]
    for arg_name in required:
        if arg_name not in func_args.keys():
            # return Difficulty.MEDIUM
            return Difficulty.HARD

    # 检查函数参数
    wrong_ratio = check_function_args(id, func_args, api_args, user_messages_content)
    # print(f"wrong_ratio: {wrong_ratio:.2f}")
    if wrong_ratio == 0:
        return Difficulty.EASY
    # elif wrong_ratio < 0.5:
    #     return Difficulty.MEDIUM
    else:
        return Difficulty.HARD

def check_functions(id, functions, apis, user_messages) -> Difficulty:
    difficulties = []
    for function in functions:
        difficulty = check_function(id, function, apis, user_messages)
        difficulties.append(difficulty)

    difficulties = sorted(difficulties, key=lambda x: x.value, reverse=True)
    return difficulties[0]

def check_data_difficulty(data) -> Difficulty:
    id = data["id"]
    user_messages = data["user_messages"]
    targets = data["targets"]
    apis = data["apis"]
    llm_responses = data["llm_responses"]

    # Check if there is no target
    if targets_is_null(targets):
        return Difficulty.HARD

    # Check if there is only one target that is answered
    # num_answer_targets = get_answer_targets_count(targets)
    # if num_answer_targets == 1:
    #     raise ValueError("Only one target can be answered in results file")

    last_target = get_last_target(targets)
    # print(f"{target=}")
    assert last_target != []
    functions = last_target
    # print(f"{functions=}")
    num_functions = len(functions)

    # 一般情况下，如果用户消息只有一条，那么这个问题很可能是多API调用的问题
    # round_a 有可能误判，不过不要紧
    if len(user_messages) == 1 and num_functions == 1:
        return Difficulty.HARD

    return check_functions(id, functions, apis, user_messages)


def classify_targets_by_difficulty(results_file):
    easy_data = []
    medium_data = []
    hard_data = []
    with open(results_file, "r") as f:
        for line in f:
            data = json.loads(line)
            difficulty = check_data_difficulty(data)
            if difficulty == Difficulty.EASY:
                easy_data.append(data)
            elif difficulty == Difficulty.MEDIUM:
                medium_data.append(data)
            elif difficulty == Difficulty.HARD:
                hard_data.append(data)
            else:
                raise ValueError(f"Unknown difficulty: {difficulty}")
    return easy_data, medium_data, hard_data

def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", type=str, default="./results_b/round_b_results-20250316_3_1.jsonl")
    parser.add_argument("--easy_data_file", type=str, default="./data/easy_data.jsonl")
    parser.add_argument("--medium_data_file", type=str, default="./data/medium_data.jsonl")
    parser.add_argument("--hard_data_file", type=str, default="./data/hard_data.jsonl")

    args = parser.parse_args()
    return args

def main():
    args = get_args()
    results_file = args.results_file

    easy_data, medium_data, hard_data = classify_data_by_difficulty(results_file)
    print(f"easy_data: {len(easy_data)}")
    print(f"medium_data: {len(medium_data)}")
    print(f"hard_data: {len(hard_data)}")

if __name__ == "__main__":
    main()