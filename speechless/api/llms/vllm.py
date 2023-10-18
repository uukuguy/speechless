# vllm/examples/offline_inference.py
import sys, os
from typing import Dict, List, AsyncIterator
from loguru import logger
import torch
from vllm import LLM, EngineArgs, LLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine


import uuid
def random_uuid() -> str:
    return str(uuid.uuid4().hex)
from .base_llm import BaseLLM

os.environ['RAY_memory_monitor_refresh_ms'] = '0'


# LLM
def create_vllm_llm(model_name_or_path, tensor_parallel_size=1, trust_remote_code=True):
    if tensor_parallel_size < 1:
        tensor_parallel_size = torch.cuda.device_count()
    llm = LLM(model=model_name_or_path, \
            trust_remote_code=True, \
            tensor_parallel_size=tensor_parallel_size
    )
    return llm

# LLMEngine
def create_vllm_engine(model_name_or_path, tensor_parallel_size=1, trust_remote_code=True):
    if tensor_parallel_size < 1:
        tensor_parallel_size = torch.cuda.device_count()

    params = dict(
        model = model_name_or_path,
        tensor_parallel_size = tensor_parallel_size,
        trust_remote_code = trust_remote_code,
    )
    engine_args = EngineArgs(**params)

    engine = LLMEngine.from_engine_args(engine_args)

    return engine

# AsyncLLMEngine
def create_vl_async_engine(model_name_or_path, tensor_parallel_size=1, trust_remote_code=True):
    if tensor_parallel_size < 1:
        tensor_parallel_size = torch.cuda.device_count()

    params = dict(
        model = model_name_or_path,
        tensor_parallel_size = tensor_parallel_size,
        trust_remote_code = trust_remote_code,
    )
    engine_args = AsyncEngineArgs(**params)

    engine = AsyncLLMEngine.from_engine_args(engine_args)

    return engine

from ..protocol.openai import CompletionParams as OpenAICompletionParams, CompletionResponse as OpenAICompletionResponse
from ..protocol.openai import ChatCompletionParams as OpenAIChatCompletionParams, ChatCompletionResponse as OpenAIChatCompletionResponse
# ==================== class VllmLLM ====================
class VllmLLM(BaseLLM):
    """
    VllmLLM Implementation
    """


    def __init__(self, settings) -> None:
        engine_args = AsyncEngineArgs(**settings.vllm_engine_params)
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)


    def generate_sampling_params_from_request(self, request_dict: Dict[str, str]):

        openai_completion_params = OpenAICompletionParams.from_request(request_dict)

        def update_attrs(target, source):
            for k, v in source.__dict__.items():
                if hasattr(target, k):
                    setattr(target, k, v)

        # -------------------- sampling_params --------------------
        # The sampling parameters for VLLM
        from ..protocol.sampling_params import VLLMSamplingParams
        sampling_params = VLLMSamplingParams()
        # Update attributes in sampling_params from openai_completion_params
        update_attrs(sampling_params, openai_completion_params)
        # Use property sampling method according to the protocol of VLLM.
        sampling_method = request_dict.pop("sampling_method", "normal")
        try:
            sampling_params.use_sampling_method(sampling_method)
        except Exception as e:
            logger.warning(f"An exception is occurred when using sampling method '{sampling_method}'. Using normal sampling. Exception: {e}")
            sampling_params.use_normal_sampling()

        return sampling_params


    # -------------------- generate() --------------------
    def generate(self, prompt: str, request_dict: Dict[str, str]) -> str:
        sampling_params = self.generate_sampling_params_from_request(request_dict)
        sampling_params_dict = sampling_params.__dict__
        vllm_sampling_params = SamplingParams(**sampling_params_dict)

        # LLM
        # outputs = self.llm.generate([prompt], sampling_params)
        # for output in outputs:
        #     prompt = output.prompt
        #     generated_text = output.outputs[0].text
        #     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        # generated_text = outputs[0].outputs[0].text

        # LLMEngine
        request_id = random_uuid()
        generated_text = ""
        request_id = random_uuid()
        self.engine.add_request(request_id, prompt, vllm_sampling_params)
        while True:
            request_outputs = self.engine.step()
            request_output = request_outputs[0]
            # print(f"{type(request_output)=}")
            # print(f"{request_output=}")
            if request_output.finished:
                generated_text = request_output.outputs[0].text
            # for request_output in request_outputs:
            #     if request_output.finished:
            #         print(request_output)

            if not self.engine.has_unfinished_requests():
                break
            # if not (self.engine.has_unfinished_requests() or test_prompts):
            #     break

        return {
            'text': generated_text,
        }
    

    async def async_generate(
        self, prompt: str, request_dict: Dict[str, str], request_id: str
    ) -> AsyncIterator[str]:
        """
        asynchronously generate text using LLM based on an input prompt
        """

        sampling_params = self.generate_sampling_params_from_request(request_dict)
        sampling_params_dict = sampling_params.__dict__

        # Beam Search
        # vllm/sampling_params.py
        # ----- Best for 34b -----
        # sampling_params.use_beam_search_sampling(n=4, best_of=4)

        # sampling_params_dict = {k: v for k, v in request_dict.items() if k in ['temperature', 'max_tokens', 'n', 'best_of', 'stop']}
        # # sampling_params_dict['stop'] = ["````"]

        # sampling_params_dict['use_beam_search'] = True
        # sampling_params_dict['n'] = 4
        # sampling_params_dict['best_of'] = 4
        # sampling_params_dict['early_stopping'] = False # must be in [True, False, 'never']
        # sampling_params_dict['temperature'] = 0.0
        # sampling_params_dict['top_p'] = 1.0
        # sampling_params_dict['top_k'] = -1

        # ------ Best for 13b -----
        # sampling_params['use_beam_search'] = True
        # sampling_params['n'] = 5
        # sampling_params['best_of'] = 5
        # sampling_params['temperature'] = 0.0
        # sampling_params['top_p'] = 1.0
        # sampling_params['top_k'] = -1
        # sampling_params['early_stopping'] = False # must be in [True, False, 'never']

        # # Greedy
        # sampling_params['use_beam_search'] = False
        # sampling_params['temperature'] = 1e-6
        # sampling_params['best_of'] = 1
        # sampling_params['top_p'] = 1.0
        # sampling_params['top_k'] = -1

        # # non_beam_search
        # sampling_params['use_beam_search'] = False
        # sampling_params['early_stopping'] = False
        # sampling_params['length_penalty'] = 1.0

        vllm_sampling_params = SamplingParams(**sampling_params_dict)
        async for request_output in self.engine.generate(prompt, vllm_sampling_params, request_id):
            generated_text = request_output.outputs[0].text
            yield {
                'text': generated_text,
            }

    async def agenerate(
        self, prompt: str, sampling_params: Dict[str, str], request_id: str
    ) -> AsyncIterator[str]:
        """
        asynchronously generate text using LLM based on an input prompt
        """
        # avoid mypy error https://github.com/python/mypy/issues/5070
        if False:  # pylint: disable=using-constant-test
            yield

    async def abort(self, request_id: str):
        """
        abort a request
        """
        self.engine.abort(request_id)

    def embeddings(self, text: str) -> List[float]:
        """
        create embeddings from the input text
        """
        pass