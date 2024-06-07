#!/usr/bin/env python

from abc import ABC, abstractmethod
from typing import Any, Optional, List
import torch


class BaseLLM(ABC):
    def __init__(self, model_name: Optional[str] = None, *args, **kwargs):
        self.model_name = model_name
        self.model = self.load_model(*args, **kwargs)


    @abstractmethod
    def load_model(self, *args, **kwargs):
        """Loads a model, that will be responsible for scoring.

        Returns:
            A model object
        """
        pass


    @abstractmethod
    def generate(self, *args, **kwargs) -> str:
        """Runs the model to output LLM response.

        Returns:
            A string.
        """
        pass


    @abstractmethod
    async def a_generate(self, *args, **kwargs) -> str:
        """Runs the model to output LLM response.

        Returns:
            A string.
        """
        pass

class VllmAIModel(BaseLLM):
    def __init__(self, model_path=None, max_tokens=2048, *args, **kwargs):
        assert model_path is not None, "模型路径不能为空"
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.chat_model = self.load_model()


    def load_model(self):
        from vllm import LLM


        return LLM(model=self.model_path, trust_remote_code=True, tensor_parallel_size=torch.cuda.device_count())


    def generate(self, prompt: str) -> str:
        from vllm import SamplingParams


        sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=self.max_tokens)
        outputs = self.chat_model.generate(prompt, sampling_params)
        generated_text = ''
        for output in outputs:
            generated_text += output.outputs[0].text
        return generated_text


    async def a_generate(self, prompt: str) -> str:
        raise NotImplementedError("异步接口尚未实现")


    def get_model_name(self):
        return self.model_name


class LlamaCppAIModel(BaseLLM):
    def __init__(self, model_path=None, n_gpu_layers=32, n_ctx=2048, max_tokens=500, *args, **kwargs):
        assert model_path is not None, "模型路径不能为空"
        self.model_name = 'llama-cpp'
        self.model_path = model_path
        self.n_gpu_layers = n_gpu_layers
        self.max_tokens = max_tokens
        self.n_ctx = n_ctx
        self.chat_model = self.load_model()


    def load_model(self):
        from llama_cpp import Llama


        return Llama(model_path=self.model_path, n_gpu_layers=self.n_gpu_layers, n_ctx=self.n_ctx)


    def generate(self, prompt: str) -> str:
        output = self.chat_model(prompt, max_tokens=self.max_tokens)
        output = output['choices'][0]['text'].strip()
        return output


    async def a_generate(self, prompt: str) -> str:
        raise NotImplementedError("异步接口尚未实现")


    def get_model_name(self):
        return self.model_name
