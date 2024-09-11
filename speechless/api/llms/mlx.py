import os
from copy import deepcopy
from typing import Dict, List, AsyncIterator, Union
import mlx_lm
from .base_llm import BaseLLM
from loguru import logger

class MlxLLM(BaseLLM):
    def __init__(self, model_path: str, eos_token: str = None, ignore_chat_template: bool = False):
        self.model_path = model_path
        self.ignore_chat_template = ignore_chat_template
        self.tokenizer_config = {"trust_remote_code": True}
        super().__init__()
        self.model, self.tokenizer = self._load_model()

        if eos_token:
            self.tokenizer_config["eos_token"] = eos_token
        else:
            self.tokenizer_config["eos_token"] = self.tokenizer.eos_token
    
    def _load_model(self):
        model, tokenizer = mlx_lm.load(
            self.model_path, 
            adapter_path=None, 
            tokenizer_config=self.tokenizer_config)
        return model, tokenizer

    # -------------------- generate() --------------------
    def generate(self, prompt_or_messages: Union[str, List[Dict[str, str]]], temperature: float = 0.7, max_new_tokens: int = 1024, top_p=1.0, min_p=0.0, verbose=False):
        orig_prompt = deepcopy(prompt)
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

        gen_kwargs = {
            'temp': temperature,
            'max_tokens': max_new_tokens,
        }

        generated_text = mlx_lm.generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            verbose=verbose,
            **gen_kwargs
        )

        # generated_text = generated_text[len(orig_prompt):]

        return {
            'text': generated_text
        }


    # -------------------- agenerate() --------------------
    async def agenerate(
        self, prompt: str, sampling_params: Dict[str, str], request_id: str
    ) -> AsyncIterator[str]:
        raise NotImplementedError("MLX LLM does not support async generation")

    # -------------------- embeddings() --------------------
    def embeddings(self, text: str) -> List[float]:
        raise NotImplementedError("MLX LLM does not support embeddings")
