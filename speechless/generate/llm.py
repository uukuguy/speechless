#!/usr/bin/env python

from abc import ABC, abstractmethod
from typing import Any, Optional, List, Union, Generator
import torch
from tqdm import tqdm


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

    def generate_batch(self, instructions: List[str], batch_size: int=2, **kw_sampling_params) -> Generator[Any, Any, Any]: 
        cached_instructions = []
        s = 0
        for i, prompt in enumerate(tqdm(instructions, ncols=100)):
            cached_instructions.append(prompt)
            if i < len(instructions) - 1 and len(cached_instructions) < batch_size:
                continue
            generated_texts = self.generate(cached_instructions, **kw_sampling_params)
            e = i + 1
            yield s, e, generated_texts
            s = e

    # def generate(self, prompt: str, template=0.8, top_p=0.95) -> str:
    def generate(self, prompts: Optional[Union[str, List[str]]] = None, **kw_sampling_params) -> Union[str, List[str]]:  
        """
        SamplingParams:
                n: int = 1,
                best_of: Optional[int] = None,
                presence_penalty: float = 0.0,
                frequency_penalty: float = 0.0,
                repetition_penalty: float = 1.0,
                temperature: float = 1.0,
                top_p: float = 1.0,
                top_k: int = -1,
                min_p: float = 0.0,
                seed: Optional[int] = None,
                use_beam_search: bool = False,
                length_penalty: float = 1.0,
                early_stopping: Union[bool, str] = False,
                stop: Optional[Union[str, List[str]]] = None,
                stop_token_ids: Optional[List[int]] = None,
                include_stop_str_in_output: bool = False,
                ignore_eos: bool = False,
                max_tokens: Optional[int] = 16,
                logprobs: Optional[int] = None,
                prompt_logprobs: Optional[int] = None,
                skip_special_tokens: bool = True,
                spaces_between_special_tokens: bool = True,
                logits_processors: Optional[List[LogitsProcessor]] = None,
        """
        from vllm import SamplingParams

        # sampling_params = SamplingParams(temperature=template, top_p=top_p, max_tokens=self.max_tokens)
        sampling_params = SamplingParams(**kw_sampling_params)
        outputs = self.chat_model.generate(prompts, sampling_params, use_tqdm=False)

        if isinstance(prompts, list):
            generated_texts = [ output.outputs[0].text for output in outputs]
            return generated_texts
        else:
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
