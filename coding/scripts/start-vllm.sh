#!/bin/bash

export MODEL=DIFC2025-round_a-SFT-Qwen3-4B/model_weights
export VLLM_PORT=8089
export TENSOR_PARALLEL_SIZE=1

vllm serve ${MODEL} \
    --served-model-name DIFC2025-round_a-SFT-Qwen3-4B \
    --host 0.0.0.0 \
    --port ${VLLM_PORT} \
    --dtype auto \
    --tensor-parallel-size ${TENSOR_PARALLEL_SIZE}