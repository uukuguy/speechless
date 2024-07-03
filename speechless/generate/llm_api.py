#!/usr/bin/env python
import os
from loguru import logger
from abc import ABC, abstractmethod

class LLM_API(ABC):
    def __init__(self, model=None):
        pass
    
    def generate(self, instruction, generate_args, system_prompt=None, verbose=False):
        assert instruction is not None
        
        direct_messages = [
            {
                "role": "system",
                "content": "你是一名AI，你需要用你的专业知识回答用户的问题。要求回答专业，准确，条理清楚。" if system_prompt is None else system_prompt,
            },
            {
                "role": "user",
                "content": instruction,
            },
        ]
        messages = direct_messages
        if verbose:
            print(f"{messages=}")

        if not isinstance(generate_args, dict):
            generate_args = {}

        try:
            response = self.client.chat.completions.create(
                model=self.model, # 填写需要调用的模型名称
                messages=messages,
                **generate_args,
            )
            if verbose:
                print(response)
            generated_text = response.choices[0].message.content
        except Exception as e:
            print(e)
            generated_text = f"Error: {e}"

        return generated_text

class ZhipuAI_API(LLM_API):
    def __init__(self, api_key=None, model=None):
        if api_key is None:
            api_key = os.getenv("ZHIPUAI_API_KEY")

        from zhipuai import ZhipuAI
        self.client = ZhipuAI(api_key=api_key) # 请填写您自己的APIKey

        if model is None:
            model = "glm-4"
        self.model = model
    
class DashScope_API(LLM_API):
    def __init__(self, model=None):
        os.environ['OPENAI_BASE_URL'] = os.getenv("DASHSCOPE_BASE_URL")
        os.environ['OPENAI_API_KEY'] = os.getenv("DASHSCOPE_API_KEY")

        from openai import OpenAI
        client =OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))

        if model is None:
            model = "qwen-long"
            # model = "qwen-plus"
            # model = "qwen-max"
            # model = "qwen-turbo"
        self.model = model
        
class OpenAI_API(LLM_API):
    def __init__(self, model=None):
        from openai import OpenAI
        self.client =OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))

        if model is None:
            model = "gpt-4o"
            # model= "claude-3-opus-20240229"
            # model= "gpt-4-turbo"
        self.model = model

def get_llm_api(LLM_API="ZhipuAI", model=None):
    if LLM_API == "ZhipuAI":
        llm_api = ZhipuAI_API(model=model)
    elif LLM_API == "DashScope":
        llm_api = DashScope_API(model=model)
    elif LLM_API == "OpenAI":
        llm_api = OpenAI_API(model=model)
    else:
        raise ValueError(f"Invalid LLM_API: {LLM_API}")

    return llm_api

def do_query(args, client, model):

    llm_api = get_llm_api(LLM_API=args.llm_api, model=args.model)

    generate_args = {
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        # "repeat_penalty": args.repeat_penalty, 
        # "stop": ["\n"],
    }
    system_prompt = "你是一名AI，你需要用你的专业知识回答用户的问题。要求回答专业，准确，条理清楚。"
    generated_text = llm_api.generate(args.question, 
                                      generate_args=generate_args, 
                                      system_prompt=system_prompt, 
                                      verbose=args.verbose)
    print(generated_text)

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="GLMLaw2024")
    parser.add_argument("--question", type=str, help="Question")
    parser.add_argument("--llm_api", type=str, default="ZhipuAI", choices=['ZhipuAI', 'OpenAI', 'DashScope'], help="LLM API")
    parser.add_argument("--model", type=str, default=None, help="Model")
    parser.add_argument("--verbose", action="store_true", help="Verbose")
    parser.add_argument("--max_tokens", type=int, default=512, help="Max tokens")
    parser.add_argument("--temperature", type=float, default=0.95, help="Temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top p")
    parser.add_argument("--repeat_penalty", type=float, default=1.1, help="Repeat penalty")
    return parser.parse_args()

def main(args):


    do_query(args, client, model)

if __name__ == "__main__":
    args = get_args()
    main(args)
