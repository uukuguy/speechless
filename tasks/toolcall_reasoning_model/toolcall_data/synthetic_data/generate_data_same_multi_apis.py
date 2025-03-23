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
        system_prompt = "你是一个对话生成器，请根据接收到的json数据，按照对话示例生成相应的多轮对话。"
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
    prompt_dialogue = """
# 核心规则说明
1. **多次调用场景**：当用户必须在一句话中通过"另外""同时""此外"等词语提出多个请求，且必须在一个`<functioncall>`中完成多次API调用。
2. **参数独立性**：每次调用的参数可共享个别参数，但不能所有参数都一致，必须包含所有required参数。
3. **响应匹配**：每个API调用对应一个`FUNCTION RESPONSE`，按调用顺序排列。
4. **格式规范**：严格遵循JSON格式，确保生成的对话数据可解析。
5. **句式多样性**：注意每轮生成句子风格多样，不要一成不变。在有共享参数时句子可以又一个主语引出多个并行的信息。
6. **以10%的概率随机增加无关api的问题**：可在多轮对话中随机增加一句无关api的问题，ASSISTAN直接回答，无需调用API。

# 对话示例（重点观察多次调用格式）
<对话示例>
api定义:
{
      "name": "query_token_balance",
      "description": "查询指定地址在特定区块链上的代币余额。",
      "parameters": {
        "type": "object",
        "properties": {
          "wallet_address": {
            "type": "string",
            "description": "钱包的区块链地址"
          },
          "token_contract_address": {
            "type": "string",
            "description": "代币的智能合约地址"
          },
          "network": {
            "type": "string",
            "description": "区块链网络名称"
          },
          "include_usd_value": {
            "type": "boolean",
            "description": "是否包含以美元计算的余额价值",
            "default": false
          },
          "decimals": {
            "type": "integer",
            "description": "代币的小数位数，用于正确显示余额"
          }
        },
        "required": [
          "wallet_address",
          "token_contract_address",
          "network"
        ]
      }
}
对话：
[
[
    {// USER单次消息触发多次调用
        "from": "USER",
        "value": "我想查一下钱包地址0x742d35Cc6634C0532925a3b844Bc454e4438f44e在Ethereum网络上，代币合约地址为0x1985365e9f78359a9B6AD760e32412f4a445E862的代币余额，并且需要包含以美元计算的余额价值，代币的小数位数是18。另外，我还想查一下钱包地址bc1qar0srrr7xfkvy5l643lydnw9re59gtzzwf5mdq在Bitcoin网络上，代币合约地址为0x6B175474E89094C44Da98b954EedeAC495271d0F的代币余额。"
    },
    {
        "from": "ASSISTANT",
        "value": "<functioncall> [
            {// ASSISTANT单次响应包含多次调用
                \"name\": \"query_token_balance\",
                \"arguments\": {
                    \"wallet_address\": \"0x742d35Cc6634C0532925a3b844Bc454e4438f44e\",
                    \"token_contract_address\": \"0x1985365e9f78359a9B6AD760e32412f4a445E862\",
                    \"network\": \"Ethereum\",
                    \"include_usd_value\": true,
                    \"decimals\": 18
                }
            },
            {
                \"name\": \"query_token_balance\",
                \"arguments\": {
                    \"wallet_address\": \"bc1qar0srrr7xfkvy5l643lydnw9re59gtzzwf5mdq\",
                    \"token_contract_address\": \"0x6B175474E89094C44Da98b954EedeAC495271d0F\",
                    \"network\": \"Bitcoin\"
                }
            }
        ]</functioncall>"
    }
    {// 多次调用的响应按顺序排列
        "from": "FUNCTION RESPONSE",
        "value": "{ \"wallet_address\": \"0x742d35Cc6634C0532925a3b844Bc454e4438f44e\", \"token_contract_address\": \"0x1985365e9f78359a9B6AD760e32412f4a445E862\", \"network\": \"Ethereum\", \"balance\": { \"value\": \"2000000000000000000000000\", \"usd_value\": 7000000, \"decimals\": 18 }}"
    },
    {
        "from": "FUNCTION RESPONSE",
        "value": "{ \"wallet_address\": \"bc1qar0srrr7xfkvy5l643lydnw9re59gtzzwf5mdq\", \"token_contract_address\": \"0x6B175474E89094C44Da98b954EedeAC495271d0F\", \"network\": \"Bitcoin\", \"balance\": { \"value\": \"0\", \"usd_value\": 0, \"decimals\": 18 }}"
    },
    {// 随机增加无关API的问题，不用每轮加这个，以10%的概率增加就行
        "from": "USER",
        "value": "小明的爸爸是谁？"
    },
    {
        "from": "ASSISTANT",
        "value": "小明的爸爸是未知的，因为“小明”是一个常见的虚构名字，用来在例子或故事中代表一个普通的人物。除非有具体的上下文提供信息，否则无法确定小明的爸爸是谁。如果这是一个具体问题，需要提供更多的背景信息才能回答。"
    }
]
]
<对话示例说明>
针对如上示例的几点说明：
（1）其中ASSISTANT回答中出现了<functioncall>, 代表要开始使用api了。<functioncall>和</functioncall>之间的信息为json的字符串表示，内容包含了要调用的api和传入参数："name"为ASSITANT要调用的api名字，"arguments"里包含了该api各传入参数的值，由于用户输入信息包含了多次调用同一api的内容，因此应包含多次调用的api和传入参数。
（2）FUNCTION RESPONSE的回答内容是api调用返回的结果
（3）FUNCTION RESPONSE之后的ASSISTANT的回答，是ASSISTANT理解FUNCTION RESPONSE之后的回答
</对话示例说明>


在理解了如上内容后，请根据如下要求生成内容。
要求：

1. api定义：
    {{ api_definition }}

2. 生成{{ num_diags }}组对话，每组对话需满足：
   - 用户表达方式差异显著（如：有的用专业术语，有的用口语化表达）
   - 必须包含用户在单次对话中给出同一api多次调用的所有参数的对话场景
   - JSON数组格式返回，不要其他描述
   - 严格保证JSON格式符合要求，字符串中的引号等必须加上转义符
   - 格式遵循<对话示例>中的对话部分，要包含USER, ASSISTANT的内容

2. 参数处理规则：
    (1). 调用的工具函数中的参数名确保与函数定义中的参数名一致
    (2). 请不要对工具函数的任何参数使用默认值、缺省值或自行猜测值，只能使用用户明确提供的值
    (3). 请不要在工具函数调用中使用值为空或""的参数
    (4). 如果用户没有明确提供，不要自行猜测工具函数中任何参数的值
    (5). 工具函数定义中非必填参数，如果用户明确提供的，工具函数调用中需要包含该参数，使用用户提供的值
    (6). 工具函数定义中非必填参数，如果用户没有明确提供，工具函数调用中不需要包含该参数
    (7). 如果用户当前的回复信息与已检测到的工具函数无关时，不要回复任何内容
    (8). 参数值如果是百分比，请使用小数表示，如：0.5
    (9). 确保从用户回复信息中提取的参数值是正确的，提取的信息不要包含任何非参数值的内容
    (10). 参数值如果是日期，请注意用户回复信息中日期的年月日的正确性。比如：结束时间今年年底，应该使用开始时间的年份。

3. 对话内容必须要包含FUNCTION RESPONSE的内容，并且这部分内容不能用占位符之类的含糊省略描述，可以想象生成合乎逻辑的api调用返回结果。

4. 以10%的概率随机增加与API调用无关的对话，并确保ASSISTAN的回答不会在该轮次调用API。

5. 对话内容连贯自然。对话过程中USER不能直接提到1中的api，是否调用api要靠ASSISTANT通过逻辑推理自行判断。

6. 对话内容为中文。


"""
    t = Template(prompt_dialogue)
    prompt = t.render(api_definition=json.dumps(api_definition, ensure_ascii=False, indent=4), num_diags=num_diags)
    
    return prompt

def validate_dialogue(dialogue, api_definition):
    required_params = api_definition["parameters"].get("required", [])
    
    # 提取所有functioncall和FUNCTION RESPONSE
    function_calls = []
    function_responses = []
    for turn in dialogue:
        try:
            if turn.get("from") == "ASSISTANT" and "<functioncall>" in turn.get("value", ""):
                function_calls.append(turn["value"])
            if turn.get("from") == "FUNCTION RESPONSE":
                function_responses.append(turn["value"])
        except (KeyError, AttributeError, TypeError):
            continue

    #print('len(function_responses)=', len(function_responses))
    # 如果没有functioncall或者FUNCTION RESPONSE数量不足，则抛出异常
    if not function_calls or len(function_responses) < 2:
        raise ValueError("functioncall 或 FUNCTION RESPONSE 数量不足")

    # 检查最后一个functioncall
    try:
        last_call = function_calls[-1]
        args_str = last_call.split("<functioncall>")[1].split("</functioncall>")[0].strip()
        args = json.loads(args_str)

        if isinstance(args, list):
            for arg in args:
                if isinstance(arg, dict):
                    arguments = arg.get("arguments", {})
                    missing = [p for p in required_params if p not in arguments]
                    if missing:
                        raise ValueError(f"缺少必填参数: {missing}")
                    for param, spec in api_definition["parameters"]["properties"].items():
                        if param in arguments:
                            param_type = spec.get("type")
                            value = arguments[param]
                            if param_type == "integer" and not isinstance(value, int):
                                raise ValueError(f"参数 {param} 类型错误，应为integer")
                            elif param_type == "boolean" and not isinstance(value, bool):
                                raise ValueError(f"参数 {param} 类型错误，应为boolean")
                            elif param_type == "string" and not isinstance(value, str):
                                raise ValueError(f"参数 {param} 类型错误，应为string")
            return True  # 验证通过
        else:
            raise ValueError("functioncall内容不是列表格式")
    except (IndexError, KeyError, json.JSONDecodeError, AttributeError) as e:
        raise ValueError(f"Functioncall格式错误: {str(e)}")


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
    api_data_list = read_jsonl("all_test_data.jsonl")
    api_data_list = api_data_list[args.start:args.end]

  
    llm_client = LLMClient(OPENAI_API_KEY, OPENAI_API_BASE_URL)

    output_file_name = f"output_v6_{args.start}_{args.end}.jsonl" 
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
                #print(f"{response_content=}")

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
                            
                           # print(output_line)
                            output_file.write(output_line + "\n")
                            output_file.flush()
                        
                        except ValueError as ve:
                            print(f"验证失败丢弃数据: {str(ve)}")
                            print("问题对话内容:", json.dumps(dialogue, indent=2, ensure_ascii=False))
                        except Exception as e:
                            pass
                            
                except json.JSONDecodeError as e:
                    print(f"JSON解析失败: {str(e)}")
                  #  print("原始响应内容:", response_content)

    print("处理完成，结果已保存到 output.jsonl")

if __name__ == "__main__":
    main()
