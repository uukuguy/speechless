#!/bin/bash

export MODEL=/opt/local/llm_models/huggingface.co/Qwen/Qwen3-4B
export VLLM_PORT=8089
export TENSOR_PARALLEL_SIZE=1

vllm serve ${MODEL} \
    --served-model-name Qwen3-4B \
    --host 0.0.0.0 \
    --port ${VLLM_PORT} \
    --dtype auto \
    --tensor-parallel-size ${TENSOR_PARALLEL_SIZE}
