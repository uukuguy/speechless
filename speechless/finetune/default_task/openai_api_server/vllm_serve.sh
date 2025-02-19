#!/bin/bash

# https://docs.vllm.ai/en/stable/serving/openai_compatible_server.html
    # best_of: Optional[int] = None
    # use_beam_search: bool = False
    # top_k: Optional[int] = None
    # min_p: Optional[float] = None
    # repetition_penalty: Optional[float] = None
    # length_penalty: float = 1.0
    # stop_token_ids: Optional[List[int]] = Field(default_factory=list)
    # include_stop_str_in_output: bool = False
    # ignore_eos: bool = False
    # min_tokens: int = 0
    # skip_special_tokens: bool = True
    # spaces_between_special_tokens: bool = True
    # truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]] = None
    # prompt_logprobs: Optional[int] = None

MODEL=${OPENAI_DEFAULT_MODEL}
PORT=16006

vllm serve ${MODEL} \
    --host 0.0.0.0 \
    --port ${PORT} \
    --dtype auto \
    --tensor-parallel-size 8