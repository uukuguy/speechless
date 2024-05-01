#!/bin/bash

MODELS_DIR=/opt/local/llm_models/huggingface.co
MODEL_PATH=${MODELS_DIR}/mlx-community/Meta-Llama-3-8B-4bit
# MODEL_PATH=${MODELS_DIR}/mlx-community/Mistral-7B-v0.2-4bit
ADAPTER_DIR=${MODEL_PATH}-adapter
ADAPTER_FILE=${ADAPTER_DIR}/adapter.npz

    # --data /opt/local/datasets/alpaca_gpt4/alpaca_gpt4_mlx_data \

mkdir -p ${ADAPTER_DIR}
python lora.py \
    --train \
    --model /opt/local/llm_models/huggingface.co/mlx-community/Meta-Llama-3-8B-4bit \
    --adapter-file ${ADAPTER_FILE} \
    --batch-size 16 \
    --lora-layers 16 \
    --iters 5000 \
    --learning-rate 0.0002 \
    --steps-per-report 1 \
    --steps-per-eval 10 \
    --val-batches 10 \
    --save-every 10 \
    --test \
    --seed 18341 \