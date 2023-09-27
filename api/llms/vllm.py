# vllm/examples/offline_inference.py
import sys, os
from typing import Dict, List, AsyncIterator
import torch
from vllm import LLM, SamplingParams
from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

import uuid
def random_uuid() -> str:
    return str(uuid.uuid4().hex)
from .base_llm import BaseLLM

os.environ['RAY_memory_monitor_refresh_ms'] = '0'

# ==================== class VllmLLM ====================
class VllmLLM(BaseLLM):
    """
    VllmLLM Implementation
    """

    def __init__(self, settings) -> None:
        params = settings.model_params or {}
        model_name_or_path = settings.setup_params["repo_id"]
        model_dir = super().get_model_dir(settings.models_dir, settings.model_family, model_name_or_path)

        setup_params = {
            k: v
            for k, v in settings.setup_params.items()
            if k not in ("repo_id", "tokenizer_repo_id", "config_params")
        }
        # self.device = params.get("device_map", "cpu")
        self.device = "cuda"

        # # LLM
        # self.llm = LLM(model=model_name_or_path, \
        #         trust_remote_code=True, \
        #         tensor_parallel_size=2,
        # )

        # # LLMEngine
        # params['model'] = model_name_or_path
        # params['tensor_parallel_size'] = 2
        # params.pop("device_map", None)
        # params.pop("trust_remote_code", None)
        # # # engine_args = EngineArgs.from_cli_args(args)
        # # engine_args = EngineArgs(**params)
        # # self.engine = LLMEngine.from_engine_args(engine_args)

        # # engine_args = AsyncEngineArgs.from_cli_args(args)
        # engine_args = AsyncEngineArgs(**params)
        # self.engine = AsyncLLMEngine.from_engine_args(engine_args)

        # AsyncEngine
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        from vllm.sampling_params import SamplingParams

        params = settings.model_params or {}
        model_name_or_path = settings.setup_params["repo_id"]

        params['model'] = model_name_or_path
        params['tensor_parallel_size'] = 2
        params.pop("device_map", None)
        params.pop("trust_remote_code", None)

        # FIXME
        # for GPTQ
        # params['dtype'] = "float16"

        engine_args = AsyncEngineArgs(**params)
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    # -------------------- generate() --------------------
    def generate(self, prompt: str, sampling_params: Dict[str, str]) -> str:
        """
        Generate text from Huggingface model using the input prompt and parameters
        n: int = 1,
        best_of: Optional[int] = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        use_beam_search: bool = False,
        stop: Union[None, str, List[str]] = None,
        ignore_eos: bool = False,
        max_tokens: int = 16,
        logprobs: Optional[int] = None,
        """
        # sampling_params['max_tokens'] = sampling_params.get('max_new_tokens', 1024)
        # sampling_params.pop('max_new_tokens', None)

        # params['temperature'] = 0.0 #params.get('temperature', 1.0)
        sampling_params = SamplingParams(**sampling_params)

        # LLM
        outputs = self.llm.generate([prompt], sampling_params)

        # for output in outputs:
        #     prompt = output.prompt
        #     generated_text = output.outputs[0].text
        #     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

        print(f"{outputs=}")

        generated_text = outputs[0].outputs[0].text

        # # LLMEngine

        # request_id = random_uuid()
        # generated_text = ""
        # request_id = random_uuid()
        # self.engine.add_request(request_id, prompt, sampling_params)
        # while True:
        #     request_outputs = self.engine.step()
        #     request_output = request_outputs[0]
        #     # print(f"{type(request_output)=}")
        #     # print(f"{request_output=}")
        #     if request_output.finished:
        #         generated_text = request_output.outputs[0].text
        #     # for request_output in request_outputs:
        #     #     if request_output.finished:
        #     #         print(request_output)

        #     if not self.engine.has_unfinished_requests():
        #         break
        #     # if not (self.engine.has_unfinished_requests() or test_prompts):
        #     #     break

        return generated_text

    async def async_generate(
        self, prompt: str, sampling_params: Dict[str, str], request_id: str
    ) -> AsyncIterator[str]:
        """
        asynchronously generate text using LLM based on an input prompt
        """

        # sampling_params['temperature'] = 0.0 #sampling_params.get('temperature', 1.0)

        # Beam Search
        # vllm/sampling_params.py
        # ----- Best for 34b -----
        sampling_params['use_beam_search'] = True
        sampling_params['n'] = 4
        sampling_params['best_of'] = 4
        sampling_params['temperature'] = 0.0
        sampling_params['top_p'] = 1.0
        sampling_params['top_k'] = -1
        sampling_params['early_stopping'] = False # must be in [True, False, 'never']
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

        sampling_params = SamplingParams(**sampling_params)
        # final_output = await self.engine.generate(prompt, sampling_params, request_id)
        async for request_output in self.engine.generate(prompt, sampling_params, request_id):
            # yield request_output
            yield request_output.outputs[0].text

        # prompt = final_output.prompt
        # text = "".join([prompt + output.text for output in final_output.outputs])
        # text = final_output.outputs[0].text

        # return text

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