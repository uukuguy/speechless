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
    --group_by_length ${GROUP_BY_LENGTH} \
    --do_train \
    --do_eval \
    --bits ${BITS} \
    --lora_r ${LORA_R} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_dropout 0.05 \
    --num_experts ${NUM_EXPERTS} \
    --topk ${TOPK} \
    --adapter_dim ${ADAPTER_DIM} \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs ${NUM_TRAIN_EPOCHS} \
    --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --evaluation_strategy steps \
    --save_strategy ${SAVE_STRATEGY} \
    ${SAVE_TOTAL_LIMIT} \
    --logging_strategy "steps" \
    --eval_dataset_size ${EVAL_DATASET_SIZE} \
    --eval_steps ${EVAL_STEPS} \
    --save_steps ${SAVE_STEPS} \
    --logging_steps 1 \
    --learning_rate ${LEARNING_RATE} \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --weight_decay 0.0 \
    --warmup_steps ${WARMUP_STEPS} \
    --max_train_samples ${MAX_TRAIN_SAMPLES} \
    --max_eval_samples ${MAX_EVAL_SAMPLES} \
    --report_to tensorboard \
    --remove_unused_columns False \
    --gradient_checkpointing True \
    --ddp_find_unused_parameters False \
    --force_remove_overlength_samples False \
    --tf32 True