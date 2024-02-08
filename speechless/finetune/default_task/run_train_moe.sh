#!/bin/bash

SCRIPT_PATH=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
PARENT_PATH=$(cd "${SCRIPT_PATH}/.." ; pwd)

source ${SCRIPT_PATH}/task.env

# torchrun --nproc_per_node=8 --nnodes=${nnodes} --master_addr ${master_addr} --master_port=4741 --node_rank ${node_rank} train_moe.py \

PYTHONPATH=${SPEECHLESS_ROOT} \
torchrun --nnodes=1 --nproc_per_node=${NUM_GPUS} \
    --master_port 29509 \
    -m speechless.finetune.moe \
    --model_name_or_path ${BASE_MODEL_PATH} \
    --dataset ${DATASET} \
    --bf16 True \
    --model_max_length ${MODEL_MAX_LENGTH} \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs ${NUM_TRAIN_EPOCHS} \
    --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --evaluation_strategy "no" \
    --save_strategy ${SAVE_STRATEGY} \
    ${SAVE_TOTAL_LIMIT} \
    --logging_strategy "steps" \
    --eval_steps ${EVAL_STEPS} \
    --save_steps ${SAVE_STEPS} \
    --logging_steps 1 \
    --learning_rate ${LEARNING_RATE} \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --weight_decay 0.0 \
    --warmup_steps ${WARMUP_STEPS} \
    --tf32 True