import os, json
from jinja2 import Template
import jsonlines
from datetime import datetime
from openai import OpenAI
from typing import List, Dict, Any
from tqdm import tqdm

from loguru import logger
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

def json_load(json_fd , ensure_ascii: bool = False, use_json_repair: bool = True) -> Any:
    if use_json_repair:
        from json_repair import repair_json
        return repair_json(json_fd=json_fd, return_objects=True, ensure_ascii=ensure_ascii)
    else:
        try:
            return json.load(json_fd)
        except json.JSONDecodeError as e:
            logger.error(f"Error: {e}")
            return None


OPENAI_API_KEY = "xxx"
OPENAI_API_BASE_URL = "https://api.openai-proxy.org/v1"

class LLMClient:
    def __init__(self, api_key, base_url):
        # 使用 Azure OpenAI 客户端初始化
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            #api_version="2024-05-01-preview",
        )


    def generate_dialogue(self, prompt):
        system_prompt = "你是一个对话生成器，请根据接收到的json数据，按照对话示例生成相应的多个api调用的对话。"
        full_prompt = f"{system_prompt} {prompt}"

        # 调用 Azure OpenAI API 生成对话
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": full_prompt}],

            temperature=0.2,
            max_tokens=10000
        )
        return response.choices[0].message.content


# 读取 JSONL 文件并解析
def read_jsonl(file_path: str) -> List[Dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        return [json_loads(line.strip()) for line in f]

# 动态生成对话内容
def generate_dialogue_content(api_definition: Dict, num_diags: int) -> str:
    prompt_dialogue = """下面是读取的jsonl数据：
{{ api_definition }}

下面是一组对话示例，对话过程中ASSISTANT使用到了给定的api:
<对话示例>
api定义:
{
    "name": "generate_password",
    "description": "Generate a random password",
    "parameters": {
        "type": "object",
        "properties": {
            "length": {
                "type": "integer",
                "description": "The length of the password"
            },
            "include_symbols": {
                "type": "boolean",
                "description": "Whether to include symbols in the password",
                "default": true
            }
        },
        "required": [
            "length"
        ]
    }
}
对话：
[
[
    {"from": "USER", "value": "I need a new password. Can you generate one for me?"},
    {"from": "ASSISTANT", "value": "Of course. How long would you like your password to be? And would you like it to include symbols?"},
    {"from": "USER", "value": " I would like it to be 12 characters long and yes, please include symbols."},
    {"from": "ASSISTANT", "value": "<functioncall> {\"name\": \"generate_password\", \"arguments\": {\"length\": 12, \"include_symbols\": true}}</functioncall>"},
]
]
<对话示例说明>
针对如上示例的几点说明：
（1）其中ASSISTANT回答中出现了<functioncall>, 代表要开始使用api了。<functioncall>和</functioncall>之间的信息为json的字符串表示，内容包含了要调用的api和传入参数："name"为ASSITANT要调用的api名字，"arguments"里包含了该api各传入参数的值
</对话示例说明>

在理解了如上内容后，请根据如下要求生成内容。
要求：

1. api定义：
    {{ api_definition }}

2. 参数处理规则：
    (0). 每轮次回复用户已获得的非空参数值，同时提醒用户缺失的必填参数名称
    (1). 不要追问用户任何可选非必填参数的值
    (2). 及时用已获取的非空参数值调用工具函数，不要增加其它未获得具体值的参数
    (3). 调用的工具函数中的参数名确保与函数定义中的参数名一致
    (4). 请不要对工具函数的任何参数使用默认值、缺省值或自行猜测值，只能使用用户明确提供的值
    (5). 请不要在工具函数调用中使用值为空或""的参数
    (6). 如果用户没有明确提供，不要自行猜测工具函数中任何参数的值
    (7). 工具函数定义中非必填参数，如果用户明确提供的，工具函数调用中需要包含该参数，使用用户提供的值
    (8). 工具函数定义中非必填参数，如果用户没有明确提供，工具函数调用中不需要包含该参数
    (9). 如果用户当前的回复信息与已检测到的工具函数无关时，不要回复任何内容
    (10). 参数值如果是百分比，请使用小数表示，如：0.5
    (11). 确保从用户回复信息中提取的参数值是正确的，提取的信息不要包含任何非参数值的内容
    (12). 参数值如果是日期，请注意用户回复信息中日期的年月日的正确性。比如：结束时间今年年底，应该使用开始时间的年份。

3. 对话质量要求：
   (2) 用户不会直接提及api名称，需要通过上下文推理

4. 生成要求：
   生成{{ num_diags }}组对话，每组对话需满足：
   - 用户表达方式差异显著（如：有的用专业术语，有的用口语化表达）
   - 必须包含至少1个参数缺失需要等待用户回复的对话场景
   - 必须包含1次用户直接提供所有参数的场景
   - JSON数组格式返回，不要其他描述
   - 严格保证JSON格式符合要求，字符串中的引号等必须加上转义符
   - 格式遵循<对话示例>中的对话部分，要包含USER, ASSISTANT的内容
   - 对随机的组对话，在多轮对话的开头或最后，随机地添加一轮对话，用户提问一个于当前api无关的问题，ASSISTANT回答，然后结束对话。

5. 对话内容连贯自然。对话过程中USER不能直接提到1中的api，是否调用api要靠ASSISTANT通过逻辑推理自行判断。
6. 在多轮对话关于指定api的最后一轮，ASSISTANT需要用已获取的非空参数调用指定api，不需要等待具备全部必填参数就可以调用。
7. 对话内容为中文。


"""
    t = Template(prompt_dialogue)
    prompt = t.render(api_definition=json.dumps(api_definition, ensure_ascii=False, indent=4), num_diags=num_diags)
    
    return prompt

# 验证生成的对话是否符合API参数要求
def validate_dialogue(dialogue, api_definition):
    required_params = api_definition["parameters"].get("required", [])
    
    # 提取所有functioncall
    function_calls = [
        turn["value"] for turn in dialogue 
        if turn["from"] == "ASSISTANT" and "<functioncall>" in turn["value"]
    ]
    
    # 如果没有functioncall直接跳过
    if not function_calls:
        return
    
    # 检查最后一个functioncall
    last_call = function_calls[-1]
    try:
        # 提取参数部分
        args_str = last_call.split("<functioncall>")[1].split("</functioncall>")[0].strip()
        args = json_loads(args_str)["arguments"]
    except (IndexError, KeyError, json.JSONDecodeError) as e:
        raise ValueError(f"Functioncall格式错误: {str(e)}")

    # # 检查必填参数
    # missing = [p for p in required_params if p not in args]
    # if missing:
    #     raise ValueError(f"缺少必填参数: {missing}")

    # 检查参数类型
    for param, spec in api_definition["parameters"]["properties"].items():
        if param in args:
            param_type = spec["type"]
            if param_type == "integer" and not isinstance(args[param], int):
                raise ValueError(f"参数 {param} 类型错误，应为integer")
            elif param_type == "boolean" and not isinstance(args[param], bool):
                raise ValueError(f"参数 {param} 类型错误，应为boolean")
            # 添加其他类型检查...

def get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=100)
    parser.add_argument("--num_diags", type=int, default=2)
    return parser.parse_args()

# 主执行逻辑
def main():
    args = get_args()
    num_diags=args.num_diags
    # api_data_list = read_jsonl("1.jsonl")
    api_data_list = read_jsonl("apis_from_fcdata.jsonl")
    api_data_list = api_data_list[args.start:args.end]

  
    llm_client = LLMClient(OPENAI_API_KEY, OPENAI_API_BASE_URL)

    output_file_name = f"output_single_api_{args.start}_{args.end}.jsonl" 
    with open(output_file_name, "w", encoding="utf-8") as output_file:
        for api_data in tqdm(api_data_list):
            # 遍历每个API
            for api in tqdm(api_data["apis"]):

                # 生成对话内容
                prompt = generate_dialogue_content(api, num_diags=num_diags)
                response = llm_client.generate_dialogue(prompt)
                #print(response)
                response = response.split("</think>")[-1].strip()
                response_content = response.replace("```json\n", "").replace("```", "")
                # print(f"{response_content=}")

                try:
                    dialogues = json_loads(response_content)
                    if not isinstance(dialogues, list):
                        raise ValueError("生成结果不是列表格式")

                    for dialogue in dialogues:
                        try:
                            # 验证对话
                            validate_dialogue(dialogue, api)
                            
                            # 写入文件
                            output_line = json.dumps({
                                "api_name": api["name"],
                                "dialogue": dialogue
                            }, ensure_ascii=False)
                            
                            output_file.write(output_line + "\n")
                            output_file.flush()
                        
                        except ValueError as ve:
                            print(f"验证失败丢弃数据: {str(ve)}")
                            print("问题对话内容:", json.dumps(dialogue, indent=2, ensure_ascii=False))
                        except Exception as e:
                            pass
                            
                except json.JSONDecodeError as e:
                    print(f"JSON解析失败: {str(e)}")
                    print("原始响应内容:", response_content)

    print("处理完成，结果已保存到 output.jsonl")

if __name__ == "__main__":
    main()
