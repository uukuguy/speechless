"""
HuggingFaceLLM Implementation
https://github.com/1b5d/llm-api/blob/main/app/llms/huggingface/huggingface.py
"""
import os
from typing import AsyncIterator, Optional, Dict, List, Tuple

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

# ==================== class HumanLLM ====================
class HuggingFaceLLM(BaseLLM):
    """
    HuggingFaceLLM Implementation
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

        config = AutoConfig.from_pretrained(
            settings.setup_params["repo_id"], **setup_params
        )

        config_params = settings.setup_params.get("config_params", {})
        config_params = config_params or {}
        for attr_name, attr_value in config_params.items():
            if hasattr(config, attr_name) and isinstance(attr_value, dict):
                attr_config = getattr(config, attr_name)
                attr_config.update(attr_value)
                attr_value = attr_config
            setattr(config, attr_name, attr_value)
        # self.device = params.get("device_map", "cpu")
        self.device = "cuda"

        self.model = AutoModelForCausalLM.from_pretrained(
            settings.setup_params["repo_id"],
            config=config,
            cache_dir=model_dir,
            **params
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.setup_params["tokenizer_repo_id"],
            return_token_type_ids=False,
            cache_dir=model_dir,
            **params
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token


    # -------------------- generate() --------------------
    def generate(self, prompt: str, sampling_params: Dict[str, str]) -> str:
        """
        Generate text from Huggingface model using the input prompt and parameters
        """
        if 'max_new_tokens' not in sampling_params:
            max_new_tokens = sampling_params.pop("max_tokens", 1024)
            sampling_params['max_new_tokens'] = max_new_tokens

        for k in ['model', 'max_tokens', 'stop', 'prompt', 'sampling_method', 'n', 'best_of']:
            sampling_params.pop(k, None)
        input_ids = self.tokenizer(
            prompt,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors="pt",
        ).to(self.device)
        gen_tokens = self.model.generate(**input_ids, pad_token_id=self.tokenizer.pad_token_id, **sampling_params)
        result = self.tokenizer.batch_decode(
            gen_tokens[:, input_ids["input_ids"].shape[1] :]
        )
        generated_text = result[0]

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