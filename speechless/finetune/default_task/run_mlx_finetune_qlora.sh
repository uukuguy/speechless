#!/bin/bash
export MLX_MODEL_DIR=/opt/local/llm_models/huggingface.co/mlx-community && \
export MODEL_PATH=${MLX_MODEL_DIR}/Qwen2.5-3B-Instruct-bf16 && \
export PROMPT_FORMAT=chatml && \
export DATASET_PATH=/opt/local/datasets/OpenO1-SFT/OpenO1-SFT-instruction-response.jsonl && \
export EVAL_DATASET_SIZE=1000 && \
python -m speechless.finetune.mlx.sft \
    --do_train \
    --model ${MODEL_PATH} \
    --prompt_format ${PROMPT_FORMAT} \
    --dataset_path ${DATASET_PATH} \
    --eval_dataset_size ${EVAL_DATASET_SIZE} \
    --train_type lora-completion-only \
    --train_batch_size 4 \
    --eval_batch_size 4 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --max_seq_length 2048 \
    --logging_steps 10 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_strategy epochs \