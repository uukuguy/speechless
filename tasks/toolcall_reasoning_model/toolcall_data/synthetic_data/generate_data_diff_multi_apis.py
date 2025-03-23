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
    prompt_dialogue = """
# 核心规则说明
1. **多API调用场景**：提供的多个apis定义包含了多个api，需要生成用户在一句话中通过"然后"、"接着"、"同时"等词语提出多个不同操作的请求，对应了不同的api调用，并在一个`<functioncall>`中完成多个不同API的调用。
2. **参数独立性**：每个API调用的参数必须完全独立，不能跨API共享参数。
3. **响应匹配**：每个API调用对应一个`FUNCTION RESPONSE`，按调用顺序严格排列。
4. **格式规范**：严格遵循JSON格式，确保生成的对话数据可解析。
5. **领域跨度**：允许跨领域混合调用（如量子计算与数学计算混合）。
6. **句式多样性**：用户消息需体现复杂操作的逻辑顺序（如：先初始化→再应用门→最后纠错）。

<对话示例>
apis定义:
[
    {
        "name": "precipitation_analysis",
        "description": "分析指定时间段和区域内的降水数据。该API可以用于气象研究和灾害预警。",
        "parameters": {
            "type": "object",
            "properties": {
                "start_time": {"type": "string", "description": "分析开始的时间，格式为YYYY-MM-DD HH:MM:SS"},
                "end_time": {"type": "string", "description": "分析结束的时间，格式为YYYY-MM-DD HH:MM:SS"},
                "area": {"type": "string", "description": "分析的区域名称或代码"},
                "data_source": {"type": "string", "description": "使用的数据源，如卫星或地面站"},
                "detail_level": {"type": "integer", "description": "分析的详细程度，1为最低，5为最高"}
            },
            "required": ["start_time", "end_time", "area"]
        }
    },
    {
        "name": "invasive_species_impact",
        "description": "评估外来入侵物种对本地生态系统的影响程度，基于物种数量和分布范围。",
        "parameters": {
            "type": "object",
            "properties": {
                "species_name": {"type": "string", "description": "外来入侵物种的名称。"},
                "local_ecosystem": {"type": "string", "description": "受影响的本地生态系统类型。"},
                "population_size": {"type": "integer", "description": "入侵物种的种群数量。"},
                "distribution_area": {"type": "float", "description": "入侵物种的分布面积，单位为平方公里。"},
                "impact_duration": {"type": "integer", "description": "入侵持续时间，单位为年。"},
                "include_economic_impact": {"type": "boolean", "description": "是否包括经济影响评估。", "default": false},
                "include_biodiversity_impact": {"type": "boolean", "description": "是否包括生物多样性影响评估。", "default": true}
            },
            "required": ["species_name", "local_ecosystem", "population_size", "distribution_area"]
        }
    }
]
对话：
[
[ 
    {// USER单次消息触发多个不同API调用
        "from": "USER",
        "value": "我需要分析以下数据：1. 从2023-09-01 00:00:00到2023-09-30 23:59:59，华北地区的降水情况，使用卫星数据，详细程度为3；2. 评估红火蚁对华北农田生态系统的影响，种群数量约500万，分布面积1200平方公里，入侵持续5年，需要包括经济影响评估。"
    },
    {// ASSISTANT单次响应包含两个不同API调用
        "from": "ASSISTANT",
        "value": "<functioncall> [
            {// 降水数据分析
                \"name\": \"precipitation_analysis\",
                \"arguments\": {
                    \"start_time\": \"2023-09-01 00:00:00\",
                    \"end_time\": \"2023-09-30 23:59:59\",
                    \"area\": \"华北地区\",
                    \"data_source\": \"卫星\",
                    \"detail_level\": 3
                }
            },
            {// 入侵物种影响评估
                \"name\": \"invasive_species_impact\",
                \"arguments\": {
                    \"species_name\": \"红火蚁\",
                    \"local_ecosystem\": \"华北农田\",
                    \"population_size\": 5000000,
                    \"distribution_area\": 1200,
                    \"impact_duration\": 5,
                    \"include_economic_impact\": true
                }
            }
        ]</functioncall>"
    },
    {// 响应按调用顺序排列
        "from": "FUNCTION RESPONSE",
        "value": "{\"start_time\":\"2023-09-01 00:00:00\", \"end_time\":\"2023-09-30 23:59:59\", \"area\":\"华北地区\", \"total_precipitation\":\"350mm\", \"max_daily_precipitation\":\"50mm\", \"data_source\":\"卫星\"}"
    },
    {
        "from": "FUNCTION RESPONSE",
        "value": "{\"species_name\":\"红火蚁\", \"local_ecosystem\":\"华北农田\", \"ecological_impact\":\"高\", \"economic_loss\":\"1.2亿元/年\", \"biodiversity_loss\":\"15%\"}"
    },
    {// 10%概率的无关问题
        "from": "USER",
        "value": "红火蚁是怎么传播到华北地区的？"
    },
    {
        "from": "ASSISTANT",
        "value": "红火蚁主要通过人类活动传播，例如运输土壤、植物或货物时无意中携带蚁群。此外，气候变化和贸易全球化也加速了其扩散。"
    }
]
]
</对话示例>

在理解了如上内容后，请根据如下要求生成内容。
要求：

1. 多个apis定义：
    {{ api_definition }}

2. 生成{{ num_diags }}组对话，每组对话需满足：
   - 用户表达方式差异显著（如：有的用专业术语，有的用口语化表达）
   - 必须包含用户在单次对话中给出多个不同api调用的所有参数的对话场景
   - JSON数组格式返回，不要其他描述
   - 严格保证JSON格式符合要求，字符串中的引号等必须加上转义符
   - 格式遵循<对话示例>中的对话部分，要包含USER, ASSISTANT的内容

3. 参数处理规则：
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

4. 对话内容必须要包含FUNCTION RESPONSE的内容，并且这部分内容不能用占位符之类的含糊省略描述，可以想象生成合乎逻辑的api调用返回结果。

5. 以10%的概率随机增加与API调用无关的对话，并确保ASSISTAN的回答不会在该轮次调用API。

6. 对话内容连贯自然。对话过程中USER不能直接提到1中的api，是否调用api要靠ASSISTANT通过逻辑推理自行判断。

7. 对话内容为中文。

"""
    t = Template(prompt_dialogue)
    prompt = t.render(api_definition=json.dumps(api_definition, ensure_ascii=False, indent=4), num_diags=num_diags)
    
    return prompt

def validate_dialogue(dialogue: List[Dict], api_definitions: List[Dict]) -> bool:
    """
    验证对话中的多API调用合法性
    :param dialogue: 单组对话数据
    :param api_definitions: API定义列表
    """
    # 构建API定义映射表 {api_name: definition}
    api_map = {api["name"]: api for api in api_definitions}
    
    # 提取所有functioncall和对应的FUNCTION RESPONSE
    function_calls = []
    function_responses = []
    
    for turn in dialogue:
        if turn.get("from") == "ASSISTANT" and "<functioncall>" in turn.get("value", ""):
            try:
                # 解析functioncall列表
                call_str = turn["value"].split("<functioncall>")[1].split("</functioncall>")[0].strip()
                calls = json.loads(call_str)
                if not isinstance(calls, list):
                    calls = [calls]
               #     raise ValueError("Functioncall必须是列表格式")
                function_calls.extend(calls)
            except (json.JSONDecodeError, IndexError) as e:
                raise ValueError(f"Functioncall解析失败: {str(e)}")
        
        if turn.get("from") == "FUNCTION RESPONSE":
            function_responses.append(turn["value"])


    # 基础校验
    if len(function_calls) != len(function_responses):
        raise ValueError(f"API调用次数({len(function_calls)})与响应次数({len(function_responses)})不匹配")

    # **丢弃 FUNCTION RESPONSE 轮次少于 2 的数据**
    if len(function_responses) < 2:
        print("警告: FUNCTION RESPONSE 轮次不足 2，跳过此对话")
        return False  # 也可以选择 `raise ValueError("FUNCTION RESPONSE 轮次不足 2")`
        
    # 逐个验证API调用
    for call in function_calls:
        api_name = call.get("name")
        if not api_name:
            raise ValueError("API调用缺少name字段")
        
        # 查找API定义
        api_def = api_map.get(api_name)
        if not api_def:
            raise ValueError(f"未定义的API名称: {api_name}")
        
        # 提取参数
        arguments = call.get("arguments", {})
        if not isinstance(arguments, dict):
            raise ValueError(f"API参数必须是字典格式: {api_name}")
        
        # 校验必填参数
        required_params = api_def["parameters"].get("required", [])
        missing = [p for p in required_params if p not in arguments]
        if missing:
            raise ValueError(f"API {api_name} 缺少必填参数: {missing}")
       
    return True


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
    api_data_list = read_jsonl("apis_from_fcdata.jsonl")
    #api_data_list = read_jsonl("all_test_data.jsonl")
    api_data_list = api_data_list[args.start:args.end]

  
    llm_client = LLMClient(OPENAI_API_KEY, OPENAI_API_BASE_URL)

    output_file_name = f"multi_apis_0316_{args.start}_{args.end}.jsonl" 
    with open(output_file_name, "w", encoding="utf-8") as output_file:
        for api_data in tqdm(api_data_list):

            # 生成对话内容
            prompt = generate_dialogue_content(api_data["apis"], num_diags=num_diags)
            #print('prompt=',prompt)
            response = llm_client.generate_dialogue(prompt)
            response = response.split("</think>")[-1].strip()
            response_content = response.replace("```json\n", "").replace("```", "")
            #print(f"{response_content=}")

            try:
                dialogues = json_loads(response_content)
                if not isinstance(dialogues, list):
                    raise ValueError("生成结果不是列表格式")

                for dialogue in dialogues:
                    try:

                        validate_dialogue(dialogue, api_data["apis"])

                        # 写入文件
                        output_line = json.dumps({
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
