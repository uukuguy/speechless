#!/bin/bash

SCRIPT_ROOT=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
PARENT_PATH=$(cd "${SCRIPT_PATH}/.." ; pwd)

source ${SCRIPT_ROOT}/task.env

MODEL_PATH=${TEST_MODEL_PATH}
# MODEL=${OPENAI_MODEL_NAME}
# VLLM_PORT=${OPENAI_API_PORT}
SGLANG_PORT=${OPENAI_API_PORT}

docker run --gpus all --rm \
    --privileged \
    --shm-size 32g \
    -p ${SGLANG_PORT}:${SGLANG_PORT} \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v /opt/local/llm_models:/opt/local/llm_models \
    --env "HF_TOKEN=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" \
    --ipc=host \
    lmsysorg/sglang:latest \
    python3 -m sglang.launch_server \
    --model-path ${MODEL_PATH} \
    --host 0.0.0.0 \
    --port ${SGLANG_PORT} \
    $@