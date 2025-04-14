#!/usr/bin/env python

from abc import ABC, abstractmethod
from typing import Any, Optional, List, Union, Generator, Dict
import torch
import time
from tqdm import tqdm
from loguru import logger


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
    def __init__(self, model_path=None, max_model_len=32768, max_tokens=2048, tensor_parallel_size=0, *args, **kwargs):
        assert model_path is not None, "模型路径不能为空"
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.max_model_len = max_model_len
        if tensor_parallel_size == 0:
            tensor_parallel_size = torch.cuda.device_count()
        self.tensor_parallel_size = tensor_parallel_size

        self.chat_model = self.load_model()

    def load_model(self):
        from vllm import LLM


        return LLM(model=self.model_path, max_model_len=self.max_model_len, trust_remote_code=True, tensor_parallel_size=self.tensor_parallel_size)

    def generate_batch(self, instructions: List[str], batch_size: int=2, **kw_sampling_params) -> Generator[Any, Any, Any]: 
        cached_instructions = []
        s = 0
        for i, prompt in enumerate(tqdm(instructions, ncols=100)):
            cached_instructions.append(prompt)
            e = i + 1
            if i < len(instructions) - 1 and len(cached_instructions) < batch_size:
                continue
            generated_texts = self.generate(cached_instructions, **kw_sampling_params)
            yield s, e, generated_texts
            cached_instructions = []
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

from transformers import AutoConfig, AutoTokenizer,AutoModelForCausalLM,pipeline,BitsAndBytesConfig
class HFAIModel(BaseLLM):
    def __init__(self, model_path=None, max_tokens=2048, bits=None, model_kwargs=None, gen_kwargs=None, *args, **kwargs):
        assert model_path is not None, "模型路径不能为空"
        self.model_path = model_path
        self.max_tokens = max_tokens

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=bits == 4,
            load_in_8bit=bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ) if bits in (4, 8) else None

        if model_kwargs is None:
            model_kwargs = {
                "torch_dtype": torch.bfloat16,
                "device_map": "auto", 
                "trust_remote_code": True,
                "quantization_config": bnb_config,
            }
            model_kwargs["attn_implementation"] = "flash_attention_2"
        self.model_kwargs = model_kwargs

        if gen_kwargs is None:
            gen_kwargs = {
                "temperature": 0.75, 
                "max_new_tokens": 2048,
                "do_sample": True,
                # "top_p": 0.9,
                "min_p": 0.1,
            }
        self.gen_kwargs = gen_kwargs

        self.model, self.tokenizer = self.load_model()


    def load_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(self.model_path,
                                                    # quantization_config=bnb_config,
                                                    **self.model_kwargs)

        model.eval()

        return model, tokenizer


    def generate(self, prompt_or_messages: Union[str, List[Dict[str, str]]], gen_kwargs=None, ignore_chat_template=False, verbose=False) -> str:
        if not ignore_chat_template and (
            hasattr(self.tokenizer, "apply_chat_template")
            and self.tokenizer.chat_template is not None
        ):
            if isinstance(prompt_or_messages, str): # legacy
                messages = [{"role": "user", "content": prompt_or_messages}]
            else:
                messages = prompt_or_messages
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            if isinstance(prompt_or_messages, str): # legacy
                prompt = prompt_or_messages
            else:
                prompt = [message['role'].upper() + "\n" + message['content'] + "\n" for message in prompt_or_messages]

        if verbose:
            logger.debug(f"{prompt=}")

        if gen_kwargs is None:
            gen_kwargs = self.gen_kwargs

        start_time = time.time()
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
        
        with torch.no_grad():
            gen_tokens = self.model.generate(**inputs, pad_token_id=self.tokenizer.pad_token_id, **gen_kwargs)
            # if verbose:
            #     logger.debug(f"{gen_tokens=}")

            # eos_token_id = self.tokenizer.eos_token_id
            # eos_token_index = torch.where(gen_tokens == eos_token_id)
            # if verbose:
            #     logger.debug(f"{eos_token_id=}, {eos_token_index=}")
            # if eos_token_index[0].numel() > 0:
            #     gen_tokens = gen_tokens[:, :eos_token_index[1][0]]
            # else:
            #     gen_tokens = gen_tokens[:, :max_new_tokens]

            # if verbose:
            #     logger.debug(f"{gen_tokens=}")

        s = inputs["input_ids"].shape[1]
        e = s + gen_kwargs["max_new_tokens"]
        generated_text = self.tokenizer.decode(gen_tokens[0, s:e], skip_special_tokens=True)

        if verbose:
            logger.debug(f"{generated_text=}")
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        tps = (gen_tokens.shape[1] - inputs["input_ids"].shape[1]) / elapsed_time

        return generated_text

    def generate_batch(self, instructions: List[str], batch_size: int=2, gen_kwargs=None, ignore_chat_template=False, verbose=False) -> Generator[Any, Any, Any]: 
        cached_instructions = []
        s = 0
        for i, prompt in enumerate(tqdm(instructions, ncols=100)):
            cached_instructions.append(prompt)
            e = i + 1
            if i < len(instructions) - 1 and len(cached_instructions) < batch_size:
                continue
            generated_texts = []
            for instruction in cached_instructions:
                generated_text = self.generate(instruction, gen_kwargs=gen_kwargs, ignore_chat_template=ignore_chat_template, verbose=verbose)
                generated_texts.append(generated_text)
            yield s, e, generated_texts

            cached_instructions = []
            s = e

    async def a_generate(self, prompt: str) -> str:
        raise NotImplementedError("异步接口尚未实现")