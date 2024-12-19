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

class LLM_API(ABC):
    @classmethod
    def from_config(name: str, model: str):
        if name == "MLX":
            from .mlx import MLX_API
            return MLX_API(model)
        elif name == "Vllm":
            from .vllm import VllmAPI
            return VllmAPI(model)
        elif name == "Ollama":
            from .ollama import OllamaAPI
            return OllamaAPI(model)
        elif name == "llamacpp":
            from .llamacpp import LlamaCppAPI
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