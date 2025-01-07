import os
from loguru import logger
from abc import ABC, abstractmethod

class LLM_API(ABC):
    @classmethod
    def from_config(name: str, model: str):
        if name == "MLX":
            from .mlx import MLX_API
            return MLX_API(model)
        elif name == "Vllm":
            return VllmAPI(model)
        elif name == "Ollama":
            return OllamaAPI(model)
        elif name == "llamacpp":
            return LlamaCppAPI(model)
        elif name == "OpenAI":
            from .llm_api import OpenAI_API
            return OpenAI_API(model)
        elif name == "ZhipuAI":
            from .llm_api import ZhipuAI_API
            return ZhipuAI_API(model)
        elif name == "DashScope":
            from .llm_api import DashScope_API
            return DashScope_API(model)
        else:
            raise ValueError(f"Unknown LLM API: {name}")

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

class OllamaAPI(LLM_API):
    def __init__(self, model=None, api_key=None):
        from ollama import Ollama
        self.client = Ollama(api_key=api_key)

    def generate(self, prompt, system_message="", gen_params=None, stream=False, ignore_chat_template=False, verbose=False):
        pass

class VllmAPI(LLM_API):
    def __init__(self, model=None):
        from vllm import LLM
        self.client = LLM(model=model)
    def generate(self, prompt, system_message="", gen_params=None, stream=False, ignore_chat_template=False, verbose=False):
        pass

class LlamaCppAPI(LLM_API):
    def __init__(self, model=None):
        from llamacpp import LlamaCpp
        self.client = LlamaCpp(model=model)
    def generate(self, prompt, system_message="", gen_params=None, stream=False, ignore_chat_template=False, verbose=False):
        pass

def ollama_make_api_call(messages, max_tokens, model_name, is_final_answer=False):
    import ollama
    for attempt in range(3):
        try:
            response = ollama.chat(
                model=model_name,
                messages=messages,
                options={
                    "num_predict": max_tokens,
                    "temperature": 0.2
                }
            )
            
            print(f"Raw API response: {response}")
            
            if 'message' not in response or 'content' not in response['message']:
                raise ValueError(f"Unexpected API response structure: {response}")
            
            content = response['message']['content']
            done_reason = response.get('done', False)
            if content:
                break
        except Exception as e:
            print(f"API call failed: {e}")
            content = f"Error: {e}"
            done_reason = False

    return content, done_reason