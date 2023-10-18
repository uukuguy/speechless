# exllamav2/examples/inference.py
import sys, os
from typing import Dict, List, AsyncIterator
import torch

from exllamav2 import(
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer,
)

from exllamav2.generator import (
    ExLlamaV2BaseGenerator,
    ExLlamaV2Sampler,
    ExLlamaV2StreamingGenerator,
)

import time

# Initialize model and cache

# model_directory =  "/opt/local/llm_models/huggingface.co/TheBloke/CodeLlama-34B-Instruct-GPTQ"
# model_directory = "/opt/local/llm_models/huggingface.co/speechlessai/speechless-codellama-dolphin-orca-platypus-13b"

# Generate some text



from .base_llm import BaseLLM

system_prompt = """You are a helpful coding assistant. Always answer as helpfully as possible."""
llama_prompt = """[INST] <<SYS>>\n<|system_prompt|>\n<</SYS>>\n\n<|user_prompt|> [/INST]"""

def split_user_prompt(user_prompt):
    sep_instruction = "### Instructions:\n"
    sep_input = "### Input:\n"
    sep_response = "### Response:\n"
    spans = user_prompt.split(sep_input)
    if len(spans) == 2:
        instruction = spans[0].replace(sep_instruction, "")
        full_input = spans[1]
    else:
        instruction = ""
        full_input = user_prompt

    # input = full_input
    # response = ""
    spans = full_input.split(sep_response)
    if len(spans) == 2:
        input = spans[0]
        response = spans[1]
    else:
        input = full_input
        response = ""

    print(f"{instruction=}\n{input=}\n{response=}\n")
    return instruction, input, response

# ==================== class ExllamaV2LLM ====================
class ExllamaV2LLM(BaseLLM):
    """
    ExllamaLLM Implementation
    """

    def __init__(self, settings) -> None:
        params = settings.model_params or {}
        model_dir = super().get_model_dir(
            settings.models_dir, settings.model_family, settings.setup_params["repo_id"]
        )

        setup_params = {
            k: v
            for k, v in settings.setup_params.items()
            if k not in ("repo_id", "tokenizer_repo_id", "config_params")
        }
        # self.device = params.get("device_map", "cpu")
        self.device = "cuda"

        model_directory = settings.setup_params["repo_id"]
        config = ExLlamaV2Config()
        config.model_dir = model_directory
        config.prepare()
        # if args.length: config.max_seq_len = args.length
        # config.rope_scale = args.rope_scale
        # config.rope_alpha = args.rope_alpha
        # config.no_flash_attn = args.no_flash_attn

        print("Loading model: " + model_directory)
        model = ExLlamaV2(config)

        # allocate 18 GB to CUDA:0 and 24 GB to CUDA:1.
        # (Call `model.load()` if using a single GPU.)
        # model.load([18, 24])

        if settings.gpu_split:
            gpu_split = [float(alloc) for alloc in settings.gpu_split.split(",")]
            model.load(gpu_split)
        else:
            model.load()

        print(" -- Loading tokenizer...")
        tokenizer = ExLlamaV2Tokenizer(config)


        # Initialize generator

        cache = ExLlamaV2Cache(model)
        # self.generator = ExLlamaV2BaseGenerator(model, cache, tokenizer)
        self.generator = ExLlamaV2StreamingGenerator(model, cache, tokenizer)

        self.model = model
        self.tokenizer = tokenizer
        self.generator.warmup()



        # config = AutoConfig.from_pretrained(
        #     settings.setup_params["repo_id"], **setup_params
        # )

        # config_params = settings.setup_params.get("config_params", {})
        # config_params = config_params or {}
        # for attr_name, attr_value in config_params.items():
        #     if hasattr(config, attr_name) and isinstance(attr_value, dict):
        #         attr_config = getattr(config, attr_name)
        #         attr_config.update(attr_value)
        #         attr_value = attr_config
        #     setattr(config, attr_name, attr_value)

        # self.model = AutoModelForCausalLM.from_pretrained(
        #     settings.setup_params["repo_id"],
        #     config=config,
        #     cache_dir=model_dir,
        #     **params
        # )
        # self.tokenizer = AutoTokenizer.from_pretrained(
        #     settings.setup_params["tokenizer_repo_id"],
        #     return_token_type_ids=False,
        #     cache_dir=model_dir,
        #     **params
        # )

        # if self.tokenizer.pad_token is None:
        #     self.tokenizer.pad_token = self.tokenizer.eos_token


    # -------------------- generate() --------------------
    def generate(self, prompt: str, sampling_params: Dict[str, str]) -> str:
        """
        Generate text from Huggingface model using the input prompt and parameters
        """
        if 'max_new_tokens' not in sampling_params:
            max_new_tokens = sampling_params.pop("max_tokens", 1024)
            sampling_params['max_new_tokens'] = max_new_tokens
        max_new_tokens = sampling_params['max_new_tokens']

        for k in ['model', 'max_tokens', 'stop']:
            sampling_params.pop(k, None)

        stop = sampling_params.get('stop', [self.tokenizer.eos_token_id])

        settings = ExLlamaV2Sampler.Settings()
        settings.temperature = sampling_params.get('temperature', 0.85)
        settings.top_k = sampling_params.get('top_k', 50)
        settings.top_p = sampling_params.get('top_p', 0.8)
        settings.token_repetition_penalty = sampling_params.get('repetition_penalty', 1.15)
        # settings.disallow_tokens(self.tokenizer, [self.tokenizer.eos_token_id])

        self.generator.set_stop_conditions(stop)

        # text = self.generator.generate_simple(prompt, settings, max_new_tokens, seed = 1234)

        # active_context = get_tokenized_context(model.config.max_seq_len - min_space_in_context)
        # active_context = torch.empty((1, 0), dtype=torch.long)

        # instruction, input, response = split_user_prompt(prompt)
        # if len(instruction) == 0:
        #     instruction = system_prompt
        # prompt = llama_prompt \
        #     .replace("<|system_prompt|>", instruction) \
        #     .replace("<|user_prompt|>", input)

        # prompt = llama_prompt \
        #     .replace("<|system_prompt|>", system_prompt) \
        #     .replace("<|user_prompt|>", prompt)

        active_context = self.tokenizer.encode(prompt, add_bos = True)
        self.generator.begin_stream(active_context, settings)

        response_tokens = 0
        generated_text = ""
        while True:
            chunk, eos, tokens = self.generator.stream()
            if len(generated_text) == 0: chunk = chunk.lstrip()
            generated_text += chunk
            response_tokens += 1

            if response_tokens >= max_new_tokens:
                break

            if eos:
                break

        return generated_text

    async def async_generate(self, prompt: str, sampling_params: Dict[str, str], request_id: str) -> str:
        generated_text = self.generate(prompt, sampling_params)
        yield generated_text


    async def agenerate(
        self, prompt: str, params: Dict[str, str], request_id: str
    ) -> AsyncIterator[str]:
        """
        asynchronously generate text using LLM based on an input prompt
        """
        # avoid mypy error https://github.com/python/mypy/issues/5070
        if False:  # pylint: disable=using-constant-test
            yield

    def embeddings(self, text: str) -> List[float]:
        """
        create embeddings from the input text
        """
        pass