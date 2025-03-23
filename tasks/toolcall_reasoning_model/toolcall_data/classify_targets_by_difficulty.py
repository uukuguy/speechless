#!/usr/bin/env python

from __future__ import annotations
import json, re, os, sys
from tqdm import tqdm
import rich
from loguru import logger
from pydantic import BaseModel

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

from typing import List, Dict
# from dataclasses import dataclass, field


class IndexMetric(BaseModel):
    id: str = None
    name: str
    value: float = 0.0
    bad_cases: List = []
    sub_metrics: Dict[str, IndexMetric] = {}
    # bad_cases: list = field(default_factory=list)
    # sub_metrics: List[IndexMetric] = field(default_factory=list)

IndexMetric.update_forward_refs()  # 仍建议显式调用以确保兼容性

# @dataclass
class NoTargetIndexMetric(IndexMetric):
    name: str = "NoTargetIndex"

# @dataclass
class FunctionNameNotExistsIndexMetric(IndexMetric):
    name: str = "FunctionNameNotExistsIndex"

# @dataclass
class ArgumentNameNotExistsIndexMetric(IndexMetric):
    name: str = "ArgumentNameNotExistsIndex"

# @dataclass
class RequiredArgumentNotExistsIndexMetric(IndexMetric):
    name: str = "RequiredArgumentNotExistsIndex"

# @dataclass
class ArgumentIndexMetric(IndexMetric):
    name: str = "ArgumentIndex"

# @dataclass
class ArgumentTypeIndexMetric(IndexMetric):
    name: str = "ArgumentTypeIndex"

# @dataclass
class ArgumentsIsNullIndexMetric(IndexMetric):
    name: str = "ArgumentsIsNullIndex"

# @dataclass
class BooleanArgumentFormatIndexMetric(IndexMetric):
    name: str = "BooleanArgumentFormatIndex"

# @dataclass
class StringArgumentFormatIndexMetric(IndexMetric):
    name: str = "StringArgumentFormatIndex"
    
# @dataclass
class NumberArgumentFormatIndexMetric(IndexMetric):
    name: str = "NumberArgumentFormatIndex"

# @dataclass
class ArrayArgumentFormatIndexMetric(IndexMetric):
    name: str = "ArrayArgumentFormatIndex"

# @dataclass
class DictArgumentFormatIndexMetric(IndexMetric):
    name: str = "DictArgumentFormatIndex"

# @dataclass
class ArgumentFormatIndexMetric(IndexMetric):
    name: str = "ArgumentFormatIndex"

class ArgumentValueIndexMetric(IndexMetric):
    name: str = "ArgumentValueIndex"

class TotalMetric(IndexMetric):
    name: str = "TotalMetric"

# class Metrics(BaseModel):
#     index_metrics: Dict[str, IndexMetric] = {}
#     # index_metrics: dict[str, IndexMetric] = field(default_factory=dict)

# def add_index_metric(metrics, index_metric, bad_cases):
#     index_metric.bad_cases.extend(bad_cases) 
#     metrics.index_metrics[index_metric.name] = index_metric

def add_sub_index_metric(sub_metrics, index_metric, bad_cases):
    # index_metric = index_metric_cls()
    index_name = index_metric.name
    index_metric = sub_metrics.get(index_name, index_metric)

    index_metric.bad_cases.extend(bad_cases)
    sub_metrics[index_name] = index_metric
    return index_metric


def check_value_type(value, value_type, metrics):
    if value_type == "string":
        return isinstance(value, str)
    elif value_type == "integer":
        return isinstance(value, int)
    elif value_type == "float":
        return isinstance(value, float) or isinstance(value, int)
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

def check_string_format(id, arg_name, arg_value, arg_description, user_messages_content, sub_metrics):
    if not check_value_type(arg_value, "string", sub_metrics):
        return False

    ok = True
    if "YYYY-MM-DD HH:MM:SS" in arg_description:
        if not re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", arg_value):
            ok = False
    elif "YYYY-MM-DDTHH:mm:ssZ" in arg_description:
        if not re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", arg_value):
            ok = False
    elif "YYYY-MM-DD" in arg_description:
        if not re.match(r"\d{4}-\d{2}-\d{2}", arg_value):
            ok = False
    elif "YYYY-MM-DD HH:MM" in arg_description:
        if not re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", arg_value):
            ok = False
    elif "YYYY-MM-DD HH:mm" in arg_description:
        if not re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", arg_value):
            ok = False
    elif "HH:MM" in arg_description:
        if not re.match(r"\d{2}:\d{2}", arg_value):
            ok = False
    elif "HH:mm" in arg_description:
        if not re.match(r"\d{2}:\d{2}", arg_value):
            ok = False
    elif "ISO 08601" in arg_description: 
        if not re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", arg_value):
            ok = False
    else:
        # if "日期" not in arg_description and "时间" not in arg_description:
        #     if arg_value not in user_messages_content:
        #         # logger.debug(f"{id=}, {arg_value=}, {user_messages_content=}")
        #         ok = False
        ok = True
    if not ok:
        bad_cases = [
            {
                "name": arg_name,
                "value": arg_value,
                "type": "string",
                "description": arg_description
            }
        ]
        add_sub_index_metric(sub_metrics, StringArgumentFormatIndexMetric(id=id, value=-1), bad_cases=bad_cases)
        return False
    return True

def check_number_format(id, arg_name, arg_value, arg_description, sub_metrics):
    ok = True
    if not check_value_type(arg_value, "number", sub_metrics):
        ok = False

    if "时间戳" in arg_description:
        if not re.match(r"\d{10}", str(arg_value)):
            ok = False

    if not ok:
        bad_cases = [
            {
                "name": arg_name,
                "value": arg_value,
                "type": "number",
                "description": arg_description
            }
        ]
        add_sub_index_metric(sub_metrics, NumberArgumentFormatIndexMetric(id=id, value=-1), bad_cases=bad_cases)
        return False

    return True

def check_boolean_format(id, arg_name, arg_value, arg_description, sub_metrics):
    ok = True
    if not check_value_type(arg_value, "boolean", sub_metrics):
        ok = False

    if not ok:
        bad_cases = [
            {
                "name": arg_name,
                "value": arg_value,
                "type": "boolean",
                "description": arg_description
            }
        ]
        add_sub_index_metric(sub_metrics, BooleanArgumentFormatIndexMetric(id=id, value=-1), bad_cases=bad_cases)
        return False
    return True

def check_format(id, arg_name, arg_value, arg_type, arg_description, user_messages_content, api_arg, argument_sub_metrics):
    flag = True
    if arg_type == "string":
        flag = check_string_format(id, arg_name, arg_value, arg_description, user_messages_content, argument_sub_metrics)
    elif arg_type == "number":
        flag = check_number_format(id, arg_name, arg_value, arg_description, argument_sub_metrics)
    elif arg_type == "boolean":
        flag = check_boolean_format(id, arg_name, arg_value, arg_description, argument_sub_metrics)

    if not flag:
        return False

    return True

def check_arg_value_in_user_messages(id, arg_name, arg_value, arg_type, arg_description, user_messages_content, sub_metries) -> float:
    ok = True
    if arg_type == "string":
        if "日期" not in arg_description and "时间" not in arg_description:
            if arg_value not in user_messages_content:
                # logger.debug(f"{id=}, {arg_value=}, {user_messages_content=}")
                ok = False
    # if arg_type in ["number", "integer", "float"]:
    #     if str(arg_value) not in user_messages_content:
    #         ok = False
    if not ok:
        bad_cases = [
            {
                "name": arg_name,
                "value": arg_value,
                "type": arg_type,
                "description": arg_description
            }
        ]
        add_sub_index_metric(sub_metries, ArgumentValueIndexMetric(id=id, value=-1), bad_cases=bad_cases)
        return False
    return True

def check_function_args(id, func_args, api_args, user_messages_content, total_metrics) -> float:
    # logger.info(user_messages_content)

    argument_index_metric = ArgumentIndexMetric(id=id, value=-1)

    # if len(func_args) == 0:
    #     logger.warning(f"ArgumentsIsNull: {id}")
    #     argument_index_metric.value = -1
    #     argument_index_metric.bad_cases.append(id)
    #     add_sub_index_metric(total_metrics.sub_metrics, ArgumentsIsNullIndexMetric(id=id, value=-1), bad_cases=[id])
    #     # add_index_metric(metrics.index_metrics, ArgumentsIsNullIndexMetric(id=id, value=-1), bad_cases=[id])

    #     # add_index_metric(metrics, argument_index_metric, bad_cases=[id])
    #     return 1.0

    sub_metrics = argument_index_metric.sub_metrics

    num_wrong = 0
    if len(func_args) == 0:
        logger.warning(f"ArgumentsIsNull: {id}")
        argument_index_metric.value = -1
        argument_index_metric.bad_cases.append(id)
        add_sub_index_metric(sub_metrics, ArgumentsIsNullIndexMetric(id=id, value=-1), bad_cases=[id])
        # add_index_metric(metrics.index_metrics, ArgumentsIsNullIndexMetric(id=id, value=-1), bad_cases=[id])

        # add_index_metric(metrics, argument_index_metric, bad_cases=[id])
        # return 1.0
        # num_wrong = 1
    else:
        flag = True
        num_wrong = 0
        for arg_name, arg_value in func_args.items():
            api_arg = api_args[arg_name]
            arg_type = api_arg["type"]
            arg_description = api_arg["description"]

            if not check_value_type(arg_value, arg_type, sub_metrics):
                bad_cases = [
                    {
                        "name": arg_name,
                        "value": arg_value,
                        "type": arg_type,
                        "description": arg_description
                    }
                ]
                add_sub_index_metric(sub_metrics, ArgumentTypeIndexMetric(id=id, value=-1), bad_cases=bad_cases)
                flag = False
                # num_wrong += 1
                # logger.debug(f"{arg_name=}, {arg_value=}, {arg_type=}, {arg_description=}")
                # continue

            argument_format_index_metric = ArgumentFormatIndexMetric(id=id, value=-1)
            if not check_format(id, arg_name, arg_value, arg_type, arg_description, user_messages_content, api_arg, argument_format_index_metric.sub_metrics):
                flag = False
                # add_sub_index_metric(sub_metrics, ArgumentTypeIndexMetric(id=id, value=-1), bad_cases=[id])
                # num_wrong += 1
                # logger.debug(f"{arg_name=}, {arg_value=}, {arg_type=}, {arg_description=}")
                # continue

            if arg_type == "array":
                if not isinstance(arg_value, list):
                    flag = False
                else:
                    if "items" in api_arg:
                        item_type = api_arg["items"]["type"]
                        for v in arg_value:
                            if not check_format(id, arg_name, v, item_type, arg_description, user_messages_content, api_arg, argument_format_index_metric.sub_metrics):
                                # logger.debug(f"{arg_name=}, {arg_value=}, {arg_type=}, {arg_description=}")
                                # num_wrong += 1 
                                bad_cases = [
                                    {
                                        "name": arg_name,
                                        "item_value": v,
                                        "type": item_type,
                                        "description": arg_description
                                    }
                                ]
                                add_sub_index_metric(argument_format_index_metric.sub_metrics, ArrayArgumentFormatIndexMetric(id=id, value=-1), bad_cases=bad_cases)
                                flag = False
                                # break

            elif arg_type == "dict":
                if not isinstance(arg_value, dict):
                    flag = False
                else:
                    for k, v in arg_value.items():
                        if not check_format(id, arg_name, v, arg_type, arg_description, user_messages_content, api_arg, argument_format_index_metric.sub_metrics):
                            # num_wrong += 1 
                            bad_cases = [
                                {
                                    "name": arg_name,
                                    "item_name": k,
                                    "item_value": v,
                                    "type": arg_type,
                                    "description": arg_description
                                }
                            ]
                            add_sub_index_metric(argument_format_index_metric.sub_metrics, DictArgumentFormatIndexMetric(id=id, value=-1), bad_cases=bad_cases)
                            # logger.debug(f"{arg_name=}, {arg_value=}, {arg_type=}, {arg_description=}")
                            flag = False
                            # break

            if len(argument_format_index_metric.sub_metrics) > 0:
                add_sub_index_metric(sub_metrics, argument_format_index_metric, bad_cases=[])

            # if not check_arg_value_in_user_messages(id, arg_name, arg_value, arg_type, arg_description, user_messages_content, sub_metrics):
            #     num_wrong += 1
            #     # logger.debug(f"{arg_name=}, {arg_value=}, {arg_type=}, {arg_description=}")
            #     continue
            if not flag:
                num_wrong += 1

    
    if len(argument_index_metric.sub_metrics) > 0:
        add_sub_index_metric(total_metrics.sub_metrics, argument_index_metric, bad_cases=[])

    if len(func_args) > 0:
        wrong_ratio = num_wrong / len(func_args) 
    else:
        wrong_ratio = 1.0
    # logger.warning(f"{num_wrong=}, {len(func_args)=}, {wrong_ratio=}")
    # if wrong_ratio >= 0.5:
    #     logger.warning(f"{func_args=}")
    #     logger.info(f"{api_args=}")

    return wrong_ratio


# -------------------- Check Function difficulty --------------------
def check_function(id, function, apis, user_messages, total_metrics) -> Difficulty:
    func_name = function["name"]
    func_args = function["arguments"]

    apis_dict = {api["name"]: api for api in apis}
    api = apis_dict[func_name]
    api_args = api["parameters"]['properties']

    user_messages_content = "\n".join([m for m in user_messages])

    highest_difficulty = Difficulty.EASY

    # 函数名不在API名中
    if func_name not in apis_dict:
        bad_cases = [
            {
                "name": func_name,
                "api_names": [api["name"] for api in apis]
            }
        ]
        add_sub_index_metric(total_metrics, FunctionNameNotExistsIndexMetric(id=id, value=-1), bad_cases=bad_cases)
        return Difficulty.HARD

    # 函数参数名不在API参数名中
    bad_cases = []
    for arg_name in func_args.keys():
        if arg_name not in api_args:
            bad_case = {
                "name": arg_name,
                "value": None,
                "func_name": func_name,
                "api_argumentts": [arg for arg in api_args]
            }
    if bad_cases:
        add_sub_index_metric(total_metrics, ArgumentNameNotExistsIndexMetric(id=id, value=-1), bad_cases=bad_cases)
        return Difficulty.HARD

    # 必填参数不全
    required = api["parameters"]["required"]
    bad_cases = []
    for arg_name in required:
        arg_type = api_args[arg_name]["type"]
        arg_description = api_args[arg_name]["description"]
        if arg_name not in func_args:
            bad_cases = [
                {
                    "name": arg_name,
                    "value": None,
                    "type": arg_type,
                    "description": arg_description
                }
            ]
    # if bad_cases: 
    #     add_sub_index_metric(total_metrics.sub_metrics, RequiredArgumentNotExistsIndexMetric(id=id, value=-1), bad_cases=bad_cases)
    #     # return Difficulty.MEDIUM
    #     highest_difficulty = Difficulty.HARD

    # 检查函数参数
    wrong_ratio = check_function_args(id, func_args, api_args, user_messages_content, total_metrics)
    # print(f"wrong_ratio: {wrong_ratio:.2f}")
    function_args_difficulty = Difficulty.EASY
    if wrong_ratio == 0:
        function_args_difficulty = Difficulty.EASY
    # elif wrong_ratio < 0.5:
    #     function_args_difficulty =Difficulty.MEDIUM
    else:
        function_args_difficulty = Difficulty.HARD

    if function_args_difficulty.value > highest_difficulty.value:
        highest_difficulty = function_args_difficulty

    return highest_difficulty

# -------------------- Check Functions difficulty --------------------
def check_functions(id, functions, apis, user_messages, total_metrics) -> Difficulty:
    difficulties = []
    for function in functions:
        difficulty = check_function(id, function, apis, user_messages, total_metrics)
        difficulties.append(difficulty)

    difficulties = sorted(difficulties, key=lambda x: x.value, reverse=True)
    return difficulties[0]


# -------------------- Check Data difficulty --------------------
def check_data_difficulty(data, total_metrics) -> Difficulty:
    id = data["id"]
    user_messages = data["user_messages"]
    targets = data["targets"]
    apis = data["apis"]
    llm_responses = data["llm_responses"]


    # Check if there is no target
    if targets_is_null(targets):
        add_sub_index_metric(total_metrics.sub_metrics, NoTargetIndexMetric(value=-1.0), bad_cases=[id])
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
    # if len(user_messages) == 1 and num_functions == 1:
    #     return Difficulty.HARD 

    difficulty = check_functions(id, functions, apis, user_messages, total_metrics) 

    return difficulty


# -------------------- Classify Data by Difficulty --------------------
def classify_targets_by_difficulty(results_file):
    easy_data = []
    medium_data = []
    hard_data = []
    all_metrics = []
    with open(results_file, "r") as f:
        num_missed = 0
        for line in f:
            data = json.loads(line)
            # metrics = Metrics()
            total_metrics = TotalMetric(id=data["id"])
            difficulty = check_data_difficulty(data, total_metrics)

            if difficulty == Difficulty.HARD and len(total_metrics.sub_metrics) == 0:
                num_missed += 1
                rich.print(json.dumps(json.loads(total_metrics.json()), ensure_ascii=False))

            all_metrics.append(total_metrics)
            if difficulty == Difficulty.EASY:
                easy_data.append(data)
            elif difficulty == Difficulty.MEDIUM:
                medium_data.append(data)
            elif difficulty == Difficulty.HARD:
                hard_data.append(data)
            else:
                raise ValueError(f"Unknown difficulty: {difficulty}")
    return easy_data, medium_data, hard_data, all_metrics, num_missed

def show_metrics(metrics):
    num_bad = 0
    for i, metric in enumerate(metrics):
        # print(json.dumps(json.loads(metric.json()), indent=2))
        if metric.sub_metrics:
            num_bad += 1
            rich.print(json.dumps(json.loads(metric.json()), ensure_ascii=False, indent=2))
        # if i > 10:
        #     break
    print(f"num_bad: {num_bad}")


# -------------------- Main --------------------
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

    # results_file = "./results_a/round_results-a_20250316_2.jsonl"
    results_file = "./results_b/round_results-b_20250316_3_1.jsonl"

    easy_data_file = results_file.replace(".jsonl", "_easy_data.jsonl")
    medium_data_file = results_file.replace(".jsonl", "_medium_data.jsonl")
    hard_data_file = results_file.replace(".jsonl", "_hard_data.jsonl")

    easy_data, medium_data, hard_data, all_metrics, num_missed = classify_targets_by_difficulty(results_file)
    show_metrics(all_metrics)

    logger.warning(f"num_missed: {num_missed}")
    print(f"easy_data: {len(easy_data)}")
    print(f"medium_data: {len(medium_data)}")
    print(f"hard_data: {len(hard_data)}")


if __name__ == "__main__":
    main()