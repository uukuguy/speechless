#!/bin/bash

MODELS_DIR=/opt/local/llm_models/huggingface.co
MODEL_PATH=${MODELS_DIR}/mlx-community/Meta-Llama-3-8B-4bit
# MODEL_PATH=${MODELS_DIR}/mlx-community/Mistral-7B-v0.2-4bit
ADAPTER_DIR=${MODEL_PATH}-adapter
ADAPTER_FILE=${ADAPTER_DIR}/adapter.npz

mkdir -p ${ADAPTER_DIR}
# python lora.py \
#     --train \
#     --model /opt/local/llm_models/huggingface.co/mlx-community/Meta-Llama-3-8B-4bit \
#     --data /opt/local/datasets/alpaca_gpt4/alpaca_gpt4_mlx_data \
#     --adapter-file ${ADAPTER_FILE} \
#     --batch-size 16 \
#     --lora-layers 32 \
#     --iters 5000 \
#     --learning-rate 0.0002 \
#     --steps-per-report 1 \
#     --steps-per-eval 10 \
#     --val-batches 10 \
#     --save-every 10 \
#     --test \
#     --seed 18341 \

    # --data-path /opt/local/datasets/HuggingFaceH4/no_robots \
    # --data-path /opt/local/datasets/alpaca_gpt4/alpaca_gpt4_mlx_data \
    # --mixed-precision bf16 \
    # --peft \
    # --optimizer adamw_bnb_8bit \

autotrain llm \
    --train \
    --trainer sft \
    --model /opt/local/llm_models/huggingface.co/mistralai/Mistral-7B-Instruct-v0.2 \
    --data-path /opt/local/datasets/HuggingFaceH4/no_robots \
    --train-split train \
    --text-column messages \
    --chat-template chatml \
    --lora_r 32 \
    --lora_alpha 64 \
    --quantization int4 \
    --lr 2e-4 \
    --optimizer adamw_torch \
    --scheduler cosine \
    --warmup_ratio 0.03 \
    --block-size 1024 \
    --batch-size 2 \
    --gradient-accumulation 4 \
    --epochs 3 \
    --model-max-length 2048 \
    --logging_steps 1 \
    --padding right \
    --project-name speechless-Mistral-7B-Instruct-v02 \
    --seed 18341
