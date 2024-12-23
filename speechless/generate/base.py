import os
from loguru import logger
from abc import ABC, abstractmethod

class OllamaAPI(LLM_API):
    def __init__(self, model=None):
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