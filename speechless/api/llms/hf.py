"""
HuggingFaceLLM Implementation
https://github.com/1b5d/llm-api/blob/main/app/llms/huggingface/huggingface.py
"""
import os, time
import torch
from typing import AsyncIterator, Optional, Dict, List, Tuple, Union
from loguru import logger

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    pipeline,
)

from transformers.generation import GenerationConfig as HFGenerationConfig

from dataclasses import dataclass
@dataclass
class GenerationConfig:
    #  > Parameters that control the generation strategy used
    do_sample: Optional[bool] = False
    num_beams: Optional[int] = 1
    num_beam_groups: Optional[int] = 1
    use_cache: Optional[bool] = True

    #  > Parameters for manipulation of the model output logits
    temperature: Optional[float] = 1.0
    top_k: Optional[int] = 50
    top_p: Optional[float] = 1.0

    # > Parameters that control the length of the output
    max_new_tokens: int = 1024 # max_length = len(prompt) + max_new_tokens
    min_new_tokens: Optional[int] = None
    early_stopping: Optional[bool] = False # [True, False, 'never']
    repetition_penalty: Optional[float] = 1.0
    length_penalty: Optional[float] = 1.0
    no_repeat_ngram_size: Optional[int] = 0
    bad_words_ids: Optional[List[List[int]]] = None
    force_word_ids: Optional[List[int]] = None
    # renormalize_logits: 
    # Whether to renormalize the  Whether to renormalize the logits after applying all the logits processors 
    # or warpers (including the custom ones). 
    # It's highly recommended to set this flag to `True` as the search algorithms suppose the score logits are normalized
    # but some logit processors or warpers break the normalization.
    renormalize_logits: Optional[bool] = False 

    # constraints 
    # Custom constraints that can be added to the generation to ensure that the output will contain the use of
    # certain tokens as defined by `Constraint` objects, in the most sensible way possible.
    # constraints: Optional[List] = None

    # sequence_bias
    # Dictionary that maps a sequence of tokens to its bias term. Positive biases increase the odds of the
    # sequence being selected, while negative biases do the opposite.
    sequence_bias: Optional[Dict[Tuple[int], float]] = None

    # guidance_scale
    # The guidance scale for classifier free guidance (CFG). CFG is enabled by setting `guidance_scale > 1`.
    # Higher guidance scale encourages the model to generate samples that are more closely linked to the input
    #  prompt, usually at the expense of poorer quality.
    guidance_scale: Optional[float] = None

    low_memory: Optional[bool] = False

    # > Parameters that define the output variables of `generate`
    num_return_sequences: Optional[int] = 1

from .base_llm import BaseLLM

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# ==================== class HFLLM ====================
class HFLLM(BaseLLM):
    """
    HuggingFace Transformers LLM Implementation
    """

    def __init__(self, model_path: str, eos_token: str = None, ignore_chat_template: bool = False) -> None:
        self.model_path = model_path
        self.ignore_chat_template = ignore_chat_template
        self.model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": device,
            "trust_remote_code": True,
        }
        self.model, self.tokenizer = self._load_model()

        # self.tokenizer_config = {"trust_remote_code": True}
        # if eos_token:
        #     self.tokenizer_config["eos_token"] = eos_token
        # else:
        #     self.tokenizer_config["eos_token"] = self.tokenizer.eos_token

    def _load_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(self.model_path, **self.model_kwargs)

        return model, tokenizer


    # -------------------- generate() --------------------
    def generate(self, prompt_or_messages: Union[str, List[Dict[str, str]]], temperature: float = 0.7, max_new_tokens: int = 1024, top_p=1.0, min_p=0.0, verbose=False) -> str:
        if not self.ignore_chat_template and (
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
        if verbose:
            logger.debug(f"{prompt=}")

        gen_kwargs = {
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "top_p": top_p,
        }
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
        e = s + max_new_tokens
        generated_text = self.tokenizer.decode(gen_tokens[0, s:e], skip_special_tokens=True)

        if verbose:
            logger.debug(f"{generated_text=}")
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        tps = (gen_tokens.shape[1] - inputs["input_ids"].shape[1]) / elapsed_time

        return {
            'text': generated_text
        }

    async def async_generate(self, prompt: str, sampling_params: Dict[str, str], request_id: str) -> str:
        generated_output = self.generate(prompt, sampling_params)
        yield generated_output

    # -------------------- agenerate() --------------------
    async def agenerate(
        self, prompt: str, sampling_params: Dict[str, str], request_id: str
    ) -> AsyncIterator[str]:
        """
        Generate text stream Huggingface model using the input prompt and parameters
        """
        if 'max_new_tokens' not in sampling_params:
            max_new_tokens = sampling_params.pop("max_tokens", 1024)
            sampling_params['max_new_tokens'] = max_new_tokens

        for k in ['model', 'max_tokens', 'stop']:
            sampling_params.pop(k, None)

        input_ids = self.tokenizer(
            prompt,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors="pt",
        ).to(self.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)
        self.model.generate(
            **input_ids,
            streamer=streamer,
            pad_token_id=self.tokenizer.pad_token_id,
            **sampling_params or None
        )
        for text in streamer:
            yield text


    # -------------------- embeddings() --------------------
    def embeddings(self, text: str) -> List[float]:
        """
        Generate embeddings using the input text
        """

        pipe = pipeline(
            "feature-extraction",
            framework="pt",
            model=self.model,
            tokenizer=self.tokenizer,
            tokenize_kwargs={
                "return_token_type_ids": False,
                "return_attention_mask": True,
            },
        )
        return pipe(text)[0][0]