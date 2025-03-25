import json, toml, yaml, re
import random
from tqdm import tqdm
from loguru import logger
from typing import Any

from speechless.finetune.dataset_utils.multi_rounds import format_chat 

def jaccard_similarity(s1, s2):
    set1 = set(s1)
    set2 = set(s2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union else 0

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

def fix_toml_functions_arguments(s):
    pattern = re.compile(r"\[(.*)\.arguments\]")
    # re.findall(pattern, s)
    s = re.sub(pattern, "[functions.arguments]", s)
    return s

def fix_toml_dict(s):
    dict_matches = re.findall(r"\{.*?\}", s, re.M | re.S)
    if len(dict_matches) > 0:
        for dict_text in dict_matches:
            # d = dict_text.replace('\\\"', '"')
            d = dict_text
            if ":" in d: 
                new_dict_text = d.replace(":", " =")
                s = s.replace(dict_text, new_dict_text)
    return s

def fix_toml_digit(s):
    pattern = re.compile(r"([^\d])(\.)(\d)")
    # print(re.findall(pattern, s))
    s = re.sub(pattern, r"\1 0\2\3", s)

    pattern = re.compile(r"(\d)(\.)([^\d])")
    # print(re.findall(pattern, s))
    s = re.sub(pattern, r"\1\.0\3", s)
    return s

def fix_toml_list(s):
    pattern = r"= \((.*?)\)"
    s = re.sub(pattern, r'= "[\1]"', s)
    return s

def fix_toml(s):
    # s = s.replace("“", "\"").replace("”", "\"").replace(",\n", "\n")

    # while "\n" in s:
        # s = s.replace("\n", "")
    # while " \"\n" in s or "\n\"" in s:
    #     s = s.replace("\"\n", "\"").replace("\n\"", "\"")

    pattern = r"( [0-9]+)\.([^0-9])"
    s = re.sub(pattern, r"\1.0\2", s)

    # pattern = "= ([^0-9\"]+)\n"
    # s = re.sub(pattern, r"= \"\1\"\n", s)

    # s = s.replace('\\', '')

    s = s.replace("==", "=")
    # s = s.replace("=\n", "= 0 \n")
    s = s.replace("[functions.arguments]", "\n[functions.arguments]")

    s = fix_toml_list(s)
    s = fix_toml_functions_arguments(s)
    s = fix_toml_dict(s)
    s = fix_toml_digit(s)

    return s


def get_fuctions_from_toml(generated_text):
    sub_texts = re.findall(r"<tool_call>(.*?)<\/tool_call>", generated_text, re.M | re.S)
    # print(sub_texts)
    functions = []
    for sub_text in sub_texts:
        sub_functions = sub_text.split("[[functions]]")
        sub_functions = [ "[[functions]]\n" + x.replace("、", "") for x in sub_functions if x.strip()]
        functions.extend(sub_functions)
    logger.debug(f"{functions=}")
    functions = [ toml.loads(s)['functions'][0] for s in functions]
    logger.info(f"{functions=}")

    return functions

def convert_sharegpt_to_chatml(raw_messages):
    chatml = []
    functioncall = None
    for m in raw_messages:
        # print(f"{m=}")
        _from = m['from']
        _value = m['value']
        if _from == 'ASSISTANT':
            if "<functioncall>" in _value:
                functioncall_str = _value.split('<functioncall>')[-1].split('</functioncall>')[0]
                try:
                    functioncall = json_loads(functioncall_str)
                except Exception as e:
                    raise Exception(f"Failed to parse functioncall: {_value=}\n{functioncall_str=}\n{raw_messages=}")
                tool_calls = []
                tool_calls.append({
                    "type": "function",
                    "function": functioncall
                })
                chatml.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": tool_calls
                })
            else:
                chatml.append({
                    "role": "assistant",
                    "content": m["value"]
                })
        elif _from == 'USER':
            chatml.append({
                "role": "user",
                "content": m["value"]
            })
        elif _from == 'FUNCTION RESPONSE':
            # assert functioncall is not None, f"Function call is None when FUNCTION RESPONSE. {raw_messages=}"
            if functioncall is None:
                print(f"Function call is None when FUNCTION RESPONSE. {raw_messages=}")
                return None
            try:
                response_value = json_loads(_value)
            except Exception as e:
                exception_message = f"Failed to parse response value: {_value=}\n{raw_messages=}"
                # raise Exception(exception_message)
                print(exception_message)
                return None
            response_value.update(functioncall['arguments'])
            tool_content = json.dumps(response_value, ensure_ascii=False)
            chatml.append({
                "role": "tool",
                "name": functioncall['name'],
                "content": tool_content
            })
    return chatml


def extract_all_apis(round_test_file, all_apis_file):
    json_data = [json_loads(line.strip()) for line in open(round_test_file).readlines()]
    all_apis = dict()
    for data in tqdm(json_data):
        apis = data['apis']
        for api in apis:
            if api['name'] not in all_apis:
                all_apis[api['name']] = api
    print(f"Total {len(all_apis)} apis.")
    sorted_all_apis = sorted(all_apis.items(), key=lambda x: x[1]['name'])
    all_apis = dict(sorted_all_apis)
    json.dump(all_apis, open(all_apis_file, 'w'), ensure_ascii=False, indent=2)
    print(f"Saved {len(all_apis)} apis to {all_apis_file}")

    return all_apis


def extract_functioncall(raw_messages):
    functioncall_str = None
    # print(f"{raw_messages=}")
    for d in raw_messages:
        if not isinstance(d, dict):
            return None
        if d['from'] == 'ASSISTANT' and '<functioncall>' in d['value']:
            functioncall_str = d['value'].split('<functioncall>')[-1].split('</functioncall>')[0]
            try:
                functioncall = json_loads(functioncall_str)
                if isinstance(functioncall, list):
                    return functioncall
                elif isinstance(functioncall, dict):
                    return [functioncall]
                else:
                    return None
            except json.JSONDecodeError:
                print(f"Error decoding {functioncall_str=}")
                return None
            # print(f"{functioncall=}")
            return functioncall
    raise ValueError(f"No functioncall found {functioncall_str=}, {raw_messages=}")


def build_apis_from_messages(raw_messages, all_apis):
    functioncall = extract_functioncall(raw_messages)
    if functioncall is None:
        return None

    all_apis_list = list(all_apis.values())
    fname_group = [functioncall['name']]
    max_functions = 4
    # Randomly select 3-4 functions (including the target function)
    n = random.randint(3, max_functions)
    while len(fname_group) < n:
        fname = random.choice(all_apis_list)['name']
        if fname not in fname_group:
            fname_group.append(fname)
    fname_group = random.sample(fname_group, len(fname_group))
    # print(f"{fname_group=}")
    selected_apis = [all_apis[fname] for fname in fname_group]

    return selected_apis

def build_apis_from_multi_apis_messages(raw_messages, all_apis):
    # print(f"{raw_messages=}")
    functioncalls = extract_functioncall(raw_messages)
    # print(f"{functioncalls=}")
    if functioncalls is None:
        return None

    all_apis_list = list(all_apis.values())
    fname_group = []
    for functioncall in functioncalls:
        # print(f"{functioncall=}")
        fname_group.append(functioncall['name'])

    max_functions = len(fname_group) + 3
    # Randomly select 3-4 functions (including the target function)
    while len(fname_group) < max_functions:
        fname = random.choice(all_apis_list)['name']
        if fname not in fname_group:
            fname_group.append(fname)
    fname_group = random.sample(fname_group, len(fname_group))
    # print(f"{fname_group=}")
    selected_apis = [all_apis[fname] for fname in fname_group]

    return selected_apis


def generate_target(test_data, llm_response):
    id = test_data['id']
    apis = test_data['apis']
    user_messages = test_data['user_messages']
    # rich.print(f"{user_messages=}")
    # rich.print(f"{llm_response=}")
    if llm_response[0]["role"] == "system":
        llm_response = llm_response[1:]
    assert len(user_messages) * 2 == len(llm_response), f"{len(user_messages)=}, {len(llm_response)=}, {user_messages=}, {llm_response=}"
    targets = []
    responses = []
    for i in range(len(user_messages)):
        query = user_messages[i]
        answer = llm_response[i*2 + 1]
        #responses.append(answer["content"])
        # print(f"{answer=}")
        answer_content = answer['content']
        target = []
        # FIXME
        answer_content = answer_content.replace("</ronics>", "</tool_call>")
        # answer_content = answer_content.replace("ronics\\\\n{", "<tool_call>\\\\n{")
        answer_content = answer_content.replace("ronics\n{", "<tool_call>\n{")
        answer_content = answer_content.replace("'", "\"").replace("]\n}", "\n}")
        answer_content = answer_content.replace("\"\"", "\"")
        if not answer_content.endswith("}}"):
            answer_content = answer_content.split("}}")[0] + "}}"
        responses.append(answer_content)

        if "<tool_call>" in answer_content:
            if "</tool_call>" in answer_content:
                tool_call_str = answer_content.split('<tool_call>')[-1].split('</tool_call>')[0]
            else:
                tool_call_str = answer_content.split('<tool_call>')[-1]

            bad_json = False
            try:
                tool_call = json_loads(tool_call_str)
                if "name" not in tool_call or "arguments" not in tool_call:
                    bad_json = True
            except json.JSONDecodeError:
                print(f"[{id}] Error decoding {tool_call_str=}")
                bad_json = True

            if not bad_json:
                # print(f"{tool_call=}")
                name = tool_call['name']
                arguments = {k:v for k, v in tool_call['arguments'].items() if v}
                target =[{ 
                    "name": name,
                    "arguments": arguments,
                }]
            # print(f"{target=}")
        targets.append(target)

    # Warn if multiple targets are found
    num_targets = sum([1 for t in targets if t != []])
    if num_targets > 1:
        # print(f"Warning: {num_targets} targets found for {id=}. {targets=}")    
        pass

    def need_drop_target(t1, t2):
        return False

        # assert t1 != []
        # print(f"{t1=}, {t2=}")
        if t2 == []:
            return False
        if t1[0]["name"] != t2[0]["name"]:
            return True
        sorted_t1 = list(t1[0]["arguments"].items()) 
        sorted_t2 = list(t2[0]["arguments"].items()) 
        # print(f"{sorted_t1=}")
        # print(f"{sorted_t2=}")
        # print(f"{(sorted_t1==sorted_t2)=}") 
        # print(f"{(sorted_t1!=sorted_t2)=}") 
        # if sorted_t1 != sorted_t2:
        if sorted(list(t1[0]["arguments"].keys())) != sorted(list(t2[0]["arguments"].keys())):
            return False
        return True
    # Remain the last target that is not None, and remove the other targets
    found_target = False
    if len(targets) > 1:
        for i in range(len(targets)-1, -1, -1):
            if found_target:
                targets[i] = []
            else:
                if targets[i] != []:
                    if need_drop_target(targets[i], targets[i-1]):
                        targets[i] = []
                    else:
                        found_target = True

    return {
        "id": id,
        "apis": apis,
        "targets": targets,
        "user_messages": user_messages,
        "llm_responses": responses,
    }, num_targets

def find_api_in_apis(api_name, apis):
    for api in apis:
        if api["name"] == api_name:
            return api
    return None

def generate_target_multi_apis(test_data, llm_response, llm_response_raw):
    id = test_data['id']
    apis = test_data['apis']
    # print(f"{apis=}")
    # raise Exception("Not implemented")
    user_messages = test_data['user_messages']
    # rich.print(f"{user_messages=}")
    # rich.print(f"{llm_response=}")
    if llm_response[0]["role"] == "system":
        llm_response = llm_response[1:]
    assert len(user_messages) * 2 == len(llm_response), f"{len(user_messages)=}, {len(llm_response)=}, {user_messages=}, {llm_response=}"
    assert len(user_messages) * 2 == len(llm_response_raw), f"{len(user_messages)=}, {len(llm_response_raw)=}, {user_messages=}, {llm_response_raw=}"
    targets = []
    responses = []
    for i in range(len(user_messages)):
        query = user_messages[i]
        answer = llm_response[i*2 + 1]
        #responses.append(answer["content"])
        # print(f"{answer=}")
        answer_content = answer['content'].strip()
        target = []
        # FIXME
        # answer_content = answer_content.replace("</ronics>", "</tool_call>")
        # # answer_content = answer_content.replace("ronics\\\\n{", "<tool_call>\\\\n{")
        # answer_content = answer_content.replace("ronics\n{", "<tool_call>\n{")
        # answer_content = answer_content.replace("'", "\"").replace("]\n}", "\n}")
        # answer_content = answer_content.replace("\"\"", "\"")
        # if not answer_content.endswith("}}"):
        #     answer_content = answer_content.split("}}")[0] + "}}"
        responses.append(answer_content)

        round_targets = []
        if len(answer_content) == 0:
            answer_raw = llm_response_raw[i*2 + 1]['content'].strip()
            if "<tool_call>" in answer_raw:
                tool_calls = []
                if "</tool_call>" in answer_raw:
                    tool_call_str = answer_raw.split('<tool_call>')[-1].split('</tool_call>')[0]
                else:
                    tool_call_str = answer_raw.split('<tool_call>')[-1]

                tool_call_str = fix_toml(tool_call_str)

                bad_toml = False
                try:
                    tool_calls = toml.loads(tool_call_str)
                except Exception as e:
                    logger.error(f"[{id}] Error decoding missing tool_call: {e=} ... {tool_call_str=}")
                    bad_toml = True

                if not bad_toml:
                    if "functions" in tool_calls:
                        tool_calls = tool_calls["functions"]
                    logger.warning(f"{id=}: Fixed missing tool_calls, {tool_calls=}")
                    if not isinstance(tool_calls, list):
                        tool_calls = [tool_calls]
                    for tool_call in tool_calls:
                        if 'name' in tool_call and 'arguments' in tool_call:
                            name = tool_call['name']
                            tool_call_arguments = tool_call['arguments']
                            if isinstance(tool_call_arguments, str):
                                tool_call_arguments = json_loads(tool_call_arguments)
                            # logger.info(f"{id=}: {name=}, {tool_call_arguments=}")
                            if isinstance(tool_call_arguments, dict):
                                arguments = {k:v for k, v in tool_call_arguments.items() if v}
                                target ={ 
                                    "name": name,
                                    "arguments": arguments,
                                }
                                logger.info(f"{id=}: Fixed missing tool_call, {target=}")
                                round_targets.append(target)
        else:
            if "<tool_call>" in answer_content:
                tool_calls = []
                if "</tool_call>" in answer_content:
                    tool_call_str = answer_content.split('<tool_call>')[-1].split('</tool_call>')[0]
                else:
                    tool_call_str = answer_content.split('<tool_call>')[-1]

                bad_json = False
                try:
                    tool_calls = json_loads(tool_call_str)
                    # if "name" not in tool_call or "arguments" not in tool_call:
                    #     bad_json = True
                except json.JSONDecodeError:
                    print(f"[{id}] Error decoding : ... {tool_call_str=}")
                    bad_json = True

                if not bad_json:
                    # print(f"{tool_calls=}")
                    if not isinstance(tool_calls, list):
                        tool_calls = [tool_calls]
                    for tool_call in tool_calls:
                        if 'name' in tool_call and 'arguments' in tool_call:
                            name = tool_call['name']
                            tool_call_arguments = tool_call['arguments']
                            if isinstance(tool_call_arguments, str):
                                tool_call_arguments = json_loads(tool_call_arguments)
                            arguments = {k:v for k, v in tool_call_arguments.items() if v}
                            target ={ 
                                "name": name,
                                "arguments": arguments,
                            }
                            round_targets.append(target)
            # print(f"{target=}")
        # logger.info(f"{id=}: {round_targets=}")
        targets.append(round_targets)

    # Warn if multiple targets are found
    num_targets = sum([1 for t in targets if t != []])
    if num_targets > 1:
        # print(f"Warning: {num_targets} targets found for {id=}. {targets=}")    
        pass

    
    # 防止最后一轮错误地重复调用api
    # 通常这种情况是最后一轮是与api无关的，比如：英文。
    if len(user_messages) >= 2:
        pattern = r"[a-zA-Z]"
        if len(re.findall(pattern, user_messages[-1], re.S | re.M)) / len(user_messages[-1]) > 0.7:
            targets[-1] = []
            prev_message = user_messages[-2]
            last_message = user_messages[-1]
            logger.info(f"Warning: Remove last ENGLISH target! {id=}")
            logger.info(f"{prev_message=}")
            logger.info(f"{last_message=}")
            # logger.info(f"{targets[-2:]}")
        else:
            if targets[-1] != [] and targets[-2] != []:
                prev_targets = targets[-2]
                last_targets = targets[-1]

                if len(prev_targets) == 1 and len(last_targets) == 1:
                    prev_target = prev_targets[0]
                    last_target = last_targets[0]
                    prev_api_name = prev_target["name"]
                    last_api_name = last_target["name"]
                    prev_api_arguments = prev_target["arguments"]
                    last_api_arguments = last_target["arguments"]
                    prev_message = user_messages[-2]
                    last_message = user_messages[-1]
                    if prev_api_name == last_api_name:
                        api_name = prev_target["name"]
                        api = find_api_in_apis(api_name, apis)
                        assert api is not None

                        # 两个api完全相同，则删除最后一个
                        # prev_api_string = "\n".join(sorted([f"{k}:{v}" for k, v in prev_api_arguments.items() ]))
                        # last_api_string = "\n".join(sorted([f"{k}:{v}" for k, v in last_api_arguments.items() ]))
                        # if prev_api_string == last_api_string:
                        #     logger.error(f"Warning: Remove same api last_target! {id=}, {api_name=}")
                        #     logger.info(f"{prev_message=}")
                        #     logger.info(f"{last_message=}")
                        #     targets[-1] = []
                        # else:
                        if True:
                            api_description = api["description"]
                            arguments = api['parameters']['properties'] 
                            arg_description = "\n".join([arg['description'] for arg_name, arg in arguments.items()])
                            api_description += "\n" + arg_description

                            prev_simularity = jaccard_similarity(prev_message, api_description)
                            # prev_messages = user_messages[:-2]
                            # prev_simularity = jaccard_similarity("\n".join(prev_messages), api_description)

                            last_simularity = jaccard_similarity(last_message, api_description)
                            # last_messages = user_messages[:-1]
                            # last_simularity = jaccard_similarity("\n".join(last_messages), api_description)

                            # if last_simularity < prev_simularity:
                            if last_simularity < prev_simularity / 4:
                            # if prev_simularity > last_simularity and prev_simularity - last_simularity > 0.1:
                                logger.error(f"Warning: Remove last_target! {id=}, {api_name=}")
                                logger.info(f"{prev_message=}, {prev_simularity=}")
                                logger.info(f"{last_message=}, {last_simularity=}")
                                targets[-1] = []
                            # else:
                            #     prev_api_arguments_content = "".join(sorted([ f"{k}:{v}" for k, v in prev_api_arguments.items() ]))
                            #     last_api_arguments_content = "".join(sorted([ f"{k}:{v}" for k, v in last_api_arguments.items() ]))
                            #     if prev_api_arguments_content == last_api_arguments_content:
                            #         logger.debug(f"Warning: Remove same api last_target! {id=}, {api_name=}")
                            #         logger.debug(f"{prev_message=}, {prev_simularity=}")
                            #         logger.debug(f"{last_message=}, {last_simularity=}")
                            #         targets[-1] = []
                    else:
                        prev_api = find_api_in_apis(prev_api_name, apis)
                        last_api = find_api_in_apis(last_api_name, apis)
                        # assert prev_api is not None
                        if prev_api is not None and last_api is not None:
                            if len(user_messages) > 2:
                                targets[-1] = []

                            else:
                                prev_api_description = prev_api["description"]

                                prev_arguments = prev_api['parameters']['properties'] 
                                prev_arg_description = "\n".join([arg['description'] for arg_name, arg in prev_arguments.items()])
                                prev_api_description += "\n" + prev_arg_description

                                prev_simularity = jaccard_similarity(prev_message, prev_api_description)
                                # prev_messages = user_messages[:-2]
                                # prev_simularity = jaccard_similarity("\n".join(prev_messages), prev_api_description)
                                    

                                # print(f"{last_api_name=}")
                                # print(f"{apis=}")
                                # assert last_api is not None

                                last_api_description = last_api["description"]
                                last_arguments = last_api['parameters']['properties'] 
                                last_arg_description = "\n".join([arg['description'] for arg_name, arg in last_arguments.items()])
                                last_api_description += "\n" + last_arg_description

                                last_simularity = jaccard_similarity(last_message, last_api_description)
                                # last_messages = user_messages[:-1]
                                # last_simularity = jaccard_similarity("\n".join(last_messages), last_api_description)

                                if prev_simularity > last_simularity:
                                    logger.warning(f"Warning: Remove last_target! {id=}, {prev_api_name=}, {last_api_name=}")
                                    logger.info(f"{prev_message=}, {prev_simularity=}") 
                                    logger.info(f"{last_message=}, {last_simularity=}")
                                    targets[-1] = []
                        else:
                            if last_api is None:
                                logger.warning(f"last_api is None! {id=}, {last_api_name=}")
                                targets[-1] = []
                            else:
                                pass
                                # logger.warning(f"prev_api or last_api is None! {id=}, {prev_api_name=}, {prev_api is None}, {last_api_name=}, {last_api is None}")


    # # 从多轮相同的api中提取最后可能遗漏的参数
    # from collections import Counter
    # most_name = None
    # function_names = []
    # for target in targets:
    #     if len(target) == 1:
    #         target_name = target[0]["name"]
    #         function_names.append(target_name)
    # if len(function_names) > 0:
    #     name, count = Counter(function_names).most_common(1)[0]
    #     if count > 1:
    #         most_name = name
    
    # if most_name is not None:
    #     most_likely_targets = []
    #     for target in targets:
    #         if len(target) == 1 and target[0]["name"] == most_name:
    #             most_likely_targets.append(target)
    #     # print(f"""{id=}, {most_likely_targets=}""")
    #     if len(most_likely_targets) > 1:
    #         last_most_likely_target = most_likely_targets[-1]
    #         # print(f"{last_most_likely_target=}")
    #         last_arguments = last_most_likely_target[0]["arguments"]
    #         assert len(last_most_likely_target) == 1, f"{last_most_likely_target=}, {most_likely_targets=}"
    #         for i in range(len(most_likely_targets)-1, -1, -1):
    #             target = most_likely_targets[i]
    #             # print(f"{target=}")
    #             assert len(target) == 1, f"{target=}, {most_likely_targets=}"
    #             arguments = target[0]["arguments"]
    #             for k, v in arguments.items():
    #                 if k not in last_arguments:
    #                     last_arguments[k] = v
    #                     logger.warning(f"""Warning: Add missing argument {k}={v} to last target! {id=}""")

    def need_drop_target(t1, t2):
        return False

        # assert t1 != []
        # print(f"{t1=}, {t2=}")
        if t2 == []:
            return False
        if t1[0]["name"] != t2[0]["name"]:
            return True
        sorted_t1 = list(t1[0]["arguments"].items()) 
        sorted_t2 = list(t2[0]["arguments"].items()) 
        # print(f"{sorted_t1=}")
        # print(f"{sorted_t2=}")
        # print(f"{(sorted_t1==sorted_t2)=}") 
        # print(f"{(sorted_t1!=sorted_t2)=}") 
        # if sorted_t1 != sorted_t2:
        if sorted(list(t1[0]["arguments"].keys())) != sorted(list(t2[0]["arguments"].keys())):
            return False
        return True
    # Remain the last target that is not None, and remove the other targets
    found_target = False
    selected_target = None
    if len(targets) > 1:
        for i in range(len(targets)-1, -1, -1):
            if found_target:
                targets[i] = []
            else:
                if targets[i] != []:
                    if need_drop_target(targets[i], targets[i-1]):
                        targets[i] = []
                    else:
                        api_name = targets[i][0]["name"]
                        api = find_api_in_apis(api_name, apis)
                        if api is not None:
                            found_target = True
                            selected_target = targets[i]
                        else:
                            targets[i] = []

    logger.warning(f"{id=}, {selected_target=}")
    # Check parameter type
    if selected_target is not None:
        api_name = selected_target[0]["name"]
        api = find_api_in_apis(api_name, apis)
        assert api is not None
        arguments = api['parameters']['properties']
        for k, v in selected_target[0]["arguments"].items():
            if k in arguments:
                arg_type = arguments[k]["type"]
                if arg_type == "array":
                    if isinstance(v, str):
                        v = v.strip()
                        logger.error(f"{id=}, Error: {k} should be list! ")
                        logger.info(f"{arguments=}")
                        logger.info(f"{selected_target=}")
                        if v.startswith("[") and v.endswith("]"):
                            try:
                                selected_target[0]["arguments"][k] = json.loads(v)
                            except json.JSONDecodeError:
                                pass
                        else:
                            selected_target[0]["arguments"][k] = [ x.strip() for x in v.split(",")]
                elif arg_type == "dict":
                    if isinstance(v, str):
                        v = v.strip()
                        logger.error(f"{id=}, Error: {k} should be dict! ")
                        logger.info(f"{arguments=}")
                        logger.info(f"{selected_target=}")
                        if v.startswith("{") and v.endswith("}"):
                            # v = v.replace('\"', '"')
                            v = v.replace("=", ":")
                            try:
                                selected_target[0]["arguments"][k] = json_loads(v)
                            except json.JSONDecodeError:
                                logger.error(f"{id=}, Error: {k=} should be dict! {v=} ")
                elif arg_type == "number":
                    if isinstance(v, str):
                        v = v.strip()
                        if not v.isdigit():
                            logger.error(f"{id=}, Error: {k} should be number! ")
                            logger.info(f"{arguments=}")
                            logger.info(f"{selected_target=}")
                            if isinstance(v, str):
                                if "." in v:
                                    selected_target[0]["arguments"][k] = float(v)
                                else:
                                    selected_target[0]["arguments"][k] = int(v)
                elif arg_type == "integer":
                    if isinstance(v, str):
                        v = v.strip()
                        selected_target[0]["arguments"][k] = int(v)
                elif arg_type == "string":
                    if not isinstance(v, str):
                        logger.error(f"{id=}, Error: {k} should be string! ")
                        logger.info(f"{arguments=}")
                        logger.info(f"{selected_target=}")
                        selected_target[0]["arguments"][k] = str(v)
                    else:
                        selected_target[0]["arguments"][k] = v.strip()
                        
            else:
                logger.warning(f"{id=}, Error: {k} not in {api_name} arguments! ")
                logger.info(f"{arguments=}")
                logger.info(f"{selected_target=}")
                selected_target[0]["arguments"].pop(k)

    # if id == "36":
    #     logger.warning(f"{targets=}")
    return {
        "id": id,
        "apis": apis,
        "targets": targets,
        "user_messages": user_messages,
        "llm_responses": responses,
    }, num_targets

def build_instruction(user_messages, tools, output_format, system_content = "", include_toolcall_example=True):
    """
    output_format: "chatml" or "json"
    """
    apis = [ {"type": "function", "function" : api} for api in tools]
    # print(f"{apis=}")

    # Build messages
    system_message = {"role": "system", "content": system_content, "tools": apis}

    messages = [system_message] + [ {"role": "user", "content": m} for m in user_messages]

    # Build instruction
    instruction = format_chat(messages, output_format=output_format, add_generation_prompt=True, include_toolcall_example=include_toolcall_example)

    return instruction

def build_response_from_functioncalls(functioncalls, output_format="toml"):
    response = "<tool_call>\n"
    if output_format == "toml":
        response += toml.dumps({
            "functions": functioncalls
        })
    elif output_format == "json":
        response += json.dumps({
            "functions": functioncalls
        }, ensure_ascii=False)
    elif output_format == "yaml":
        response += yaml.dump({
            "functions": functioncalls
        }, default_flow_style=False)
    else:
        raise Exception(f"Unsupported output_format: {output_format}")
    response += "</tool_call>\n"

    return response

def toolcall_to_functions(toolcall):
    functions = []
    if "<tool_call>" in toolcall:
        functions_str = toolcall.split('<tool_call>')[-1].split('</tool_call>')[0]
        functions_str = functions_str.strip()

        if functions_str.startswith("{"):
            try:
                functions = json_loads(functions_str)
            except json.JSONDecodeError:
                logger.warning(f"""Error decoding JSON {functions_str=}""")
        else:
            try:
                functions = toml.loads(functions_str)['functions']
            except Exception as e:
                logger.warning(f"""Error decoding ToML {functions_str=}""")

    return functions