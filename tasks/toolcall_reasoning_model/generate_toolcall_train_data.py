#!/usr/bin/env python
import os
import json, toml, yaml, re
from tqdm import tqdm
from loguru import logger
import random
random.seed(10042)

from speechless.finetune.dataset_utils.multi_rounds import format_chat
from utils import json_loads, convert_sharegpt_to_chatml, extract_all_apis, build_apis_from_multi_apis_messages, build_instruction, build_response_from_functioncalls

# ----------------- Extract ToolLearning APIs -----------------
def extract_toollearning_apis(toollearning_train_file, toollearning_apis_file):
    toollearning_train_data = [json.loads(line) for line in open(toollearning_train_file).readlines()]
    print(len(toollearning_train_data), toollearning_train_data[0])

    toollearning_apis = {}
    for data in tqdm(toollearning_train_data, desc="ToolLearning"):
        functions = data['functions']
        for function in functions:
            name = function['name']
            toollearning_apis[name] = function
            toollearning_apis[name] = function
    json.dump(toollearning_apis, open(toollearning_apis_file, 'w'), ensure_ascii=False, indent=2)
    print(f"Saved {len(toollearning_apis)} apis to {toollearning_apis_file}")

    return toollearning_apis


def generate_dialogs_from_raw_data(raw_data, all_apis, function_calling_dialogs_file):
    # function_calling_dialogs_file = f"{source_data_dir}/function_calling_dialogs_v6_0228.jsonl"
    with open(function_calling_dialogs_file, 'w') as fd:
        for example in tqdm(raw_data):
            raw_messages = example['messages']
            try:
                selected_apis = build_apis_from_multi_apis_messages(raw_messages, all_apis)
            except Exception as e:
                print(f"Failed to extract apis: {e}, {raw_messages}")
                continue
            if selected_apis is None:
                continue
            apis = [ {"type": "function", "function" : api} for api in selected_apis]
            # print(f"{apis=}")
            system_message = {"role": "system", "content": "", "tools": apis}
            messages = convert_sharegpt_to_chatml(raw_messages)
            if messages is None:
                continue

            messages.insert(0, system_message)

            data = {"messages": json.dumps(messages, ensure_ascii=False)}
            fd.write(json.dumps(data, ensure_ascii=False) + '\n')

            # print(f"{messages=}")
    print(f"Saved {len(raw_data)} dialogs to {function_calling_dialogs_file}")


def generate_function_calling_sft_data(synthetic_data_files, function_calling_instructions_file, all_apis, to_prompt=True, output_format="toml"):
    """
    output_format: toml, json, yaml
    """
    multi_apis_raw_data = []
    num_no_functioncall = 0
    for data_file in synthetic_data_files:
        for line in tqdm(open(data_file).readlines()):
            # if "<functioncall>" not in line or "</functioncall>" not in line:
            if "<functioncall>" not in line:
                num_no_functioncall += 1
                continue
            try:
                json_data = json.loads(line.strip())
            except Exception as e:
                print(f"Failed to parse: {line}")
                continue
            multi_apis_raw_data.append({"messages": json_data["dialogue"]})
    print(f"{num_no_functioncall=}, {len(multi_apis_raw_data)=}")


    total_saved = 0
    with open(function_calling_instructions_file, 'w') as fd:
        for example in tqdm(multi_apis_raw_data):
            raw_messages = example['messages']
            # try:
            if True:
                selected_apis = build_apis_from_multi_apis_messages(raw_messages, all_apis)
            # except Exception as e:
            #     print(f"Failed to build_apis_from_multi_apis_messages: {e=}, {raw_messages=}")
            #     # print(json.dumps(raw_messages, ensure_ascii=False, indent=2))
            #     # raise Exception(f"Failed to extract apis: {e}, {json.dumps(raw_messages, ensure_ascii=False, indent=2)}")
            #     continue
            if selected_apis is None:
                print(f"Failed to extract apis: {selected_apis=}, {raw_messages=}")
                continue
            apis = [ {"type": "function", "function" : api} for api in selected_apis]
            # print(f"{apis=}")

            # Build messages
            system_message = {"role": "system", "content": "", "tools": apis}

            user_messages = []
            selected_raw_messages = []
            try:
                for m in raw_messages:
                    # print(f"{m=}")
                    _from = m['from']
                    _value = m.get('value')
                    if _from == "ASSISTANT": # Must the first process assistant's message
                        selected_raw_messages.append({"from": "ASSISTANT", "value": _value})
                        if "<functioncall>" in _value:
                            if random.random() < 0.9:
                                break
                    elif _from == 'USER':
                        user_messages.append({"role": "user", "content": _value})
                        selected_raw_messages.append({"from": "USER", "value": _value})
            except Exception as e:
                print(f"Failed to extract user messages: {e}, {raw_messages=}")
                # print(json.dumps(raw_messages, ensure_ascii=False, indent=2))
                # raise Exception(f"Failed to extract user messages: {e}, {raw_messages=}")
                continue

            # print(f"{len(user_messages)=}, {user_messages=}")

            messages = [system_message] + user_messages
            # print(f"{messages=}")

            if len(messages) == 0:
                continue

            if not to_prompt:
                data = {"messages": json.dumps(messages, ensure_ascii=False)}
                fd.write(json.dumps(data, ensure_ascii=False) + '\n')
            else:
                # Build instruction
                instruction = format_chat(messages, output_format=output_format, add_generation_prompt=True)

                # Build response
                response = ""
                bad_functioncall = False
                n = 0
                for m in selected_raw_messages:
                    _from = m['from']
                    if _from == "ASSISTANT":
                        # print(f"{n=}, {m=}")
                        _value = m.get('value', "")
                        if "<functioncall>" in _value:
                            # print(f"found {_value=}")
                            functioncall_str = _value.split('<functioncall>')[-1].split('</functioncall>')[0]
                            try:
                                functioncalls = json.loads(functioncall_str)
                                response = build_response_from_functioncalls(functioncalls, output_format=output_format)
                                # print(f"Found api: {response=}")

                            except Exception as e:
                                bad_functioncall = True
                                raise Exception(f"Failed to parse functioncall: {_value=}\n{functioncall_str=}\n{m=}")
                        else:
                            # print(f"{m=}")
                            response = "本轮次不需要调用函数"
                        # n += 1

                    # print(f"{n=}, {m=}, {response=}")
                    # if n  >= len(user_messages):
                    #     break

                if not bad_functioncall:
                    total_saved += 1
                    data = {"instruction": instruction, "response": response}
                    fd.write(json.dumps(data, ensure_ascii=False) + '\n')

    print(f"Saved {total_saved} instructions to {function_calling_instructions_file}")


def generate_toollearning_sft_data(toollearning_train_file, function_calling_instructions_file, to_prompt=True, output_format="toml"):
    toollearning_train_data = [json.loads(line) for line in open(toollearning_train_file).readlines()]
    len(toollearning_train_data), toollearning_train_data[0]

    with open(function_calling_instructions_file, 'w') as fd:
        for data in tqdm(toollearning_train_data, desc="ToolLearning"):
            functions = data['functions']
            apis = [ {"type": "function", "function" : api} for api in functions]

            system_message = {"role": "system", "content": "", "tools": apis}

            chatrounds = data['chatrounds']
            user_messages = []
            response = ""
            for round in chatrounds:
                role = round['role']
                content = round['content']
                if role == "user":
                    user_messages.append({"role": "user", "content": content})
                elif role == "assistant":
                    if "function_call" in round:
                        function_call = round['function_call']
                        function_call['arguments'] = json.loads(function_call['arguments'])
                        functioncalls = {"functions": [function_call]}
                        response = build_response_from_functioncalls(functioncalls, output_format=output_format)
                        break
                    else:
                        response = "本轮次不需要调用函数"

            messages = [system_message] + user_messages
            instruction = format_chat(messages, output_format="toml", add_generation_prompt=True)

            if to_prompt:
                data = {"instruction": instruction, "response": response}
                fd.write(json.dumps(data, ensure_ascii=False) + '\n')
            else:
                data = {"messages": json.dumps(messages, ensure_ascii=False)}
                fd.write(json.dumps(data, ensure_ascii=False) + '\n')

    print(f"Saved {len(toollearning_train_data)} instructions to {function_calling_instructions_file}")


def generate_rft_grpo_train_data(sft_train_file, rft_train_file):
    sft_train_data = [json.loads(line) for line in open(sft_train_file).readlines()]
    sft_train_data = random.sample(sft_train_data, len(sft_train_data))
    with open(rft_train_file, 'w', encoding="utf-8") as fd:
        for data in tqdm(sft_train_data):
            instruction = data['instruction']
            response = data['response']
            # rich.print(f"{instruction=}, {response=}")

            user_messages = re.findall(r"<\|im_start\|>user(.*?)<\|im_end\|>", instruction, re.DOTALL)
            user_messages = [m.strip() for m in user_messages]

            tools_str = re.findall(r'<tools>(.*?)</tools>', instruction, re.DOTALL)[0]
            # print(f"{len(tools_str)=}, {tools_str=}")
            tools = json_loads(tools_str.strip())
            # print(tools)
            apis = { tool['function']['name']: tool['function'] for tool in tools}
            # rich.print(apis)

            true_target = [[]]
            if "<tool_call>" in response:
                function_call = response.split("<tool_call>")[-1].split("</tool_call>")[0]
                function_call = toml.loads(function_call)
                if 'functions' in function_call:
                    functions = function_call['functions']
                else:
                    functions = function_call
                true_target = [functions]
            user_messages = json.dumps(user_messages, ensure_ascii=False)
            true_target = json.dumps(true_target, ensure_ascii=False)
            apis = json.dumps(apis, ensure_ascii=False)
            fd.write(json.dumps({"prompt": instruction, "user_messages": user_messages, "targets": true_target, "apis": apis}, ensure_ascii=False) + '\n')
    print(f"Saved {len(sft_train_data)} instructions to {rft_train_file}")


def build_round_test_rft_grpo_train_data(round_test_file, rft_train_file):

    round_test_data = [json.loads(line) for line in open(round_test_file).readlines()]
    round_test_data = random.sample(round_test_data, len(round_test_data))
    with open(rft_train_file, 'w', encoding="utf-8") as fd:
        for data in tqdm(round_test_data):
            user_messages = data['user_messages']
            apis = data['apis']

            user_messages = [m.strip() for m in user_messages]
            chat_user_messages = [{"role": "user", "content": m} for m in user_messages]

            instruction = build_instruction(chat_user_messages, apis, output_format="toml")
            apis = { api['name']: api for api in apis}
            true_target = [None]

            user_messages = json.dumps(user_messages, ensure_ascii=False)
            true_target = json.dumps(true_target, ensure_ascii=False)
            apis = json.dumps(apis, ensure_ascii=False)
            fd.write(json.dumps({"prompt": instruction, "user_messages": user_messages, "targets": true_target, "apis": apis}, ensure_ascii=False) + '\n')

    print(f"Saved {len(round_test_data)} instructions to {rft_train_file}")


def main():
    data_dir = "function_calling_data"
    source_data_dir = f"{data_dir}/source_data"
    os.makedirs(source_data_dir, exist_ok=True)

    # ----------------- Extract All APIs -----------------
    round_a_test_file = f"{data_dir}/round_a_test.jsonl"
    round_b_test_file = f"{data_dir}/round_b_test.jsonl"
    toollearning_train_file = "/opt/local/datasets/toollearning/train_data/fcdata_zh_train_v1.jsonl"

    round_a_all_apis_file = f"{data_dir}/round_a_all_apis.jsonl"
    round_b_all_apis_file = f"{data_dir}/round_b_all_apis.jsonl"
    toollearning_apis_file = f"{data_dir}/toollearning_apis.json"

    round_a_all_apis = extract_all_apis(round_test_file=round_a_test_file, all_apis_file=round_a_all_apis_file)
    round_b_all_apis = extract_all_apis(round_test_file=round_b_test_file, all_apis_file=round_b_all_apis_file)
    toollearning_apis = extract_toollearning_apis(toollearning_train_file, toollearning_apis_file)

    all_apis = {}
    all_apis.update(round_a_all_apis)
    all_apis.update(round_b_all_apis)
    all_apis.update(toollearning_apis)


    # ----------------- Generate Single API SFT Data -----------------
    # ${source_data_dir}/output_v6.jsonl -> ${source_data_dir}/function_calling_instructions_v6_0314_toml.jsonl
    # single_api_raw_file = f"{source_data_dir}/output_v6.jsonl"
    # function_calling_instructions_file = f"{source_data_dir}/function_calling_instructions_v6_0314_toml.jsonl"
    # generate_single_api_sft_data(single_api_raw_file, function_calling_instructions_file, all_apis)


    # ----------------- Generate Multi APIs SFT Data -----------------
    # multi_apis_0315
    # synthetic_data_files = [f"{source_data_dir}/output_v6_1_500.jsonl", f"{source_data_dir}/output_v6_501_970.jsonl"]
    # multi_apis_0316
    # synthetic_data_files = [f"{source_data_dir}/output_v6_1_500.jsonl", f"{source_data_dir}/output_v6_501_970.jsonl", f"{source_data_dir}/output_0315merge.jsonl"]
    # multi_apis_0316_2
    # synthetic_data_files = [f"{source_data_dir}/output_v6_1_500.jsonl", f"{source_data_dir}/output_v6_501_970.jsonl", f"{source_data_dir}/multi_apis_0316.jsonl"]
    # multi_apis_0317
    # synthetic_data_files = [f"{source_data_dir}/multi_apis_0317_from_fcdata.jsonl"] # debug
    # synthetic_data_files = [f"{source_data_dir}/output_v6_1_500.jsonl", f"{source_data_dir}/output_v6_501_970.jsonl", f"{source_data_dir}/multi_apis_0316.jsonl", f"{source_data_dir}/multi_apis_0317_from_fcdata.jsonl"]
    # multi_apis_0318
    # debug
    # synthetic_data_files = [f"{source_data_dir}/multi_apis_0318.jsonl"]
    # synthetic_data_files = [f"${source_data_dir}/output_v6_1_500.jsonl", f"{source_data_dir}/output_v6_501_970.jsonl", f"{source_data_dir}/multi_apis_0316.jsonl", f"{source_data_dir}/multi_apis_0318.jsonl"]
    # synthetic_data_files = [f"{source_data_dir}/output_v6_501_970.jsonl"]
    # synthetic_data_files = [f"{source_data_dir}/output_v6_501_970.jsonl", f"{source_data_dir}/multi_apis_0316.jsonl", f"{source_data_dir}/multi_apis_0318.jsonl"]

    # multi_apis_0316_2_toml
    synthetic_data_files = [f"{source_data_dir}/output_v6_1_500.jsonl", f"{source_data_dir}/output_v6_501_970.jsonl", f"{source_data_dir}/multi_apis_0316.jsonl"]

    # function_calling_instructions_file = f'{source_data_dir}/function_calling_instructions_multi_apis_0315.jsonl'
    # function_calling_instructions_file = f'{source_data_dir}/function_calling_instructions_multi_apis_0316.jsonl'
    # function_calling_instructions_file = f'{source_data_dir}/function_calling_instructions_multi_apis_0316_2.jsonl'
    # function_calling_instructions_file = f'{source_data_dir}/function_calling_instructions_multi_apis_0317.jsonl'
    # function_calling_instructions_file = f'{source_data_dir}/function_calling_instructions_multi_apis_0318.jsonl'
    function_calling_instructions_file = f"{data_dir}/function_calling_instructions_multi_apis_0316_2_toml.jsonl"

    generate_function_calling_sft_data(synthetic_data_files, function_calling_instructions_file, all_apis, to_prompt=True, output_format="toml")


    # ----------------- Generate ToolLearning SFT Data -----------------
    toollearning_train_file = "/opt/local/datasets/toollearning/train_data/fcdata_zh_train_v1.jsonl"
    # function_calling_instructions_file = f"${source_data_dir}/function_calling_instructions_toollearning_zh_v1.jsonl"
    function_calling_instructions_file = f"{data_dir}/function_calling_instructions_toollearning_zh_v1_toml.jsonl"

    generate_toollearning_sft_data(toollearning_train_file, function_calling_instructions_file, to_prompt=True, output_format="toml")


    # ----------------- Generate RFT GRPO Train Data -----------------
    sft_train_file = f"{data_dir}/function_calling_instructions_multi_apis_0316_2_toml.jsonl"
    rft_train_file = sft_train_file.replace("function_calling_instructions", "rft_grpo_train_data") # f"{data_dir}/rft_grpo_train_data_0316_2.jsonl"

    generate_rft_grpo_train_data(sft_train_file, rft_train_file)


    # ----------------- Generate Round Test RFT GRPO Train Data -----------------
    # round_test_file = f"${data_dir}/round_a_test.jsonl"
    # rft_train_file = f"${data_dir}/rft_grpo_train_data_round_a_toml.jsonl"
    round_test_file = f"{data_dir}/round_b_test.jsonl"
    rft_train_file = f"{data_dir}/rft_grpo_train_data_round_b_toml.jsonl"

    build_round_test_rft_grpo_train_data(round_test_file, rft_train_file)

if __name__ == "__main__":
    main()