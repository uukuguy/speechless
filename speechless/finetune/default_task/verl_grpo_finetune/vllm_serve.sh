#!/bin/bash
# https://docs.vllm.ai/en/stable/serving/openai_compatible_server.html    # best_of: Optional[int] = None    # use_beam_search: bool = False    # top_k: Optional[int] = None    # min_p: Optional[float] = None    # repetition_penalty: Optional[float] = None    # length_penalty: float = 1.0    # stop_token_ids: Optional[List[int]] = Field(default_factory=list)    # include_stop_str_in_output: bool = False    # ignore_eos: bool = False    # min_tokens: int = 0
    # skip_special_tokens: bool = True
    # spaces_between_special_tokens: bool = True    # truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]] = None
    # prompt_logprobs: Optional[int] = None

SCRIPT_ROOT=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
PARENT_PATH=$(cd "${SCRIPT_PATH}/.." ; pwd)
source ${SCRIPT_ROOT}/task.env
MODEL=${TEST_MODEL_PATH}MODEL_NAME=$(basename ${MODEL})# MODEL=${OPENAI_MODEL_NAME}
# VLLM_PORT=${OPENAI_API_PORT}

vllm serve ${MODEL} \
    --served-model-name ${MODEL_NAME} \
    --host 0.0.0.0 \
    --port ${VLLM_PORT} \
    --dtype auto \
    --enable-auto-tool-choice \
    --tool-call-parser llama3_json \
    --tensor-parallel-size ${TENSOR_PARALLEL_SIZE}

