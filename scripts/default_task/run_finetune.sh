#!/bin/bash
# From speechless-code-mistral-7b-v1.0
SCRIPT_PATH=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
PARENT_PATH=$(cd "${SCRIPT_PATH}/.." ; pwd)

source ${SCRIPT_PATH}/task.env

PYTHONPATH=${PWD}/../.. \
torchrun --nnodes=1 --nproc_per_node=${NUM_GPUS} \
    ../../finetune.py \
    ${DEEPSPEED_STAGE2} \
    --task_name ${TASK_NAME} \
    --run_name $(date +%Y%m%d-%H%M%S) \
    --model_name_or_path ${BASE_MODEL_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs ${NUM_TRAIN_EPOCHS} \
    --data_seed 10042 \
    --save_strategy steps \
    --save_total_limit 1 \
    --evaluation_strategy steps \
    --eval_dataset_size ${EVAL_DATASET_SIZE} \
    --save_steps 100 \
    --eval_steps 100 \
    --warmup_steps 20 \
    --max_eval_samples 200 \
    --dataloader_num_workers 3 \
    --logging_strategy steps \
    --logging_steps 1 \
    --report_to tensorboard \
    --remove_unused_columns False \
    --do_train \
    --max_memory_MB ${MAX_MEMORY_MB} \
    --bits 4 \
    --lora_r ${LORA_R} \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_modules all \
    --double_quant \
    --quant_type nf4 \
    --bf16 \
    --dataset ${DATASET} \
    --dataset_format ${DATASET_FORMAT} \
    --max_new_tokens 4096 \
    --model_max_len 4096 \
    --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --per_device_eval_batch_size 1 \
    --learning_rate ${LEARNING_RATE} \
    --lr_scheduler_type cosine \
    --weight_decay 0.0 \
    --seed 10042 \
    --optim paged_adamw_8bit \
    --gradient_checkpointing True \
    --group_by_length ${GROUP_BY_LENGTH} \
    --ddp_find_unused_parameters False \
    --force_remove_overlength_samples False \
    --flash_attention True \
    --rerope False \
    --repeat_steps 0