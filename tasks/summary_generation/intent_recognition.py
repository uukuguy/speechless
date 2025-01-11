INTENT_LABELS = ["concept", "status", "comparison", "timeline"]

# 这是一个示例 Prompt，可做少样本，也可再添加更多示例
# 注意: 对于中文输入，可在role=system部分写清楚语言偏好
FEW_SHOT_PROMPT = """\
你是一个熟知学术综述类型的助手，需要根据用户的提问或指令，将其意图分为以下类别之一：
1. "concept": 用户想对某个技术概念做全面调研综述
2. "status": 用户想了解某个研究方向的现状和挑战
3. "comparison": 用户想对多种方法进行对比分析
4. "timeline": 用户想了解一个技术的发展脉络

以下是一些示例：

示例输入: "请对损失函数做一个全面的调研综述，包括定义、分类以及在深度学习中的应用场景。"
示例输出: "concept"

示例输入: "Text2SQL研究现状如何，面临哪些挑战？"
示例输出: "status"

示例输入: "有哪些方法可以提升大模型的规划能力，各自优劣是什么？"
示例输出: "comparison"

示例输入: "多模态大模型的技术发展路线是什么样的？"
示例输出: "timeline"

现在请阅读用户输入，并只输出最合适的类别标签，不要额外内容。
"""

from llm_utils import LLMClient
def classify_intent_with_prompt(llm_client: LLMClient, user_input: str) -> str:
    prompt = FEW_SHOT_PROMPT + f"\n用户输入: \"{user_input}\"\n请输出类别标签:"

    generated_text = llm_client.generate(prompt, system_prompt="你是一个帮助进行学术综述意图分类的专家。")
    generated_text = generated_text.strip().lower()

    for label in INTENT_LABELS:
        if label in generated_text:
            return label
    return None

    # """
    # 使用 ChatCompletion + Few-shot Prompt 对用户输入进行意图分类。
    # 返回值：'concept' / 'direction' / 'comparison' / 'evolution' / 'general'
    # """
    # messages = [
    #     {"role": "system", "content": "你是一个帮助进行学术综述意图分类的专家。"},
    #     {"role": "user", "content": FEW_SHOT_PROMPT},
    #     {"role": "user", "content": f"用户输入: \"{user_input}\"\n请输出类别标签:"},
    # ]

    # import openai
    # response = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo",  # 或其他可用的 ChatGPT 或 GPT-4 等模型
    #     messages=messages,
    #     temperature=0.0  # 分类任务不需要过多随机性
    # )
    # # 大模型返回的消息
    # content = response["choices"][0]["message"]["content"].strip()
    # # 通常 content 会是类似于 "concept"、"direction" 等
    # # 为了安全，可以做一个小清洗
    # lower_content = content.lower()
    # for label in INTENT_LABELS:
    #     if label in lower_content:
    #         return label
    # return "general"