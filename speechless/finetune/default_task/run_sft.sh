#!/bin/bash
# From speechless-code-mistral-7b-v1.0
SCRIPT_PATH=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
PARENT_PATH=$(cd "${SCRIPT_PATH}/.." ; pwd)

source ${SCRIPT_PATH}/task.env

min=20000
max=30000
MASTER_PORT=$((RANDOM % (max - min + 1) + min))
MASTER_ADDR=$(hostname)
export OMP_NUM_THREADS=1
export TORCH_DISTRIBUTED_DEBUG=info
export NCCL_IB_DISABLE=1

PYTHONPATH=${SPEECHLESS_ROOT:-${HOME}/sandbox/LLM/speechless.ai/speechless} \
WANDB_PROJECT=${TASK_NAME} \
torchrun --nnodes=1 --nproc_per_node=${NUM_GPUS} --node_rank-0 \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT} \
    -m speechless.finetune.finetune \
    ${DEEPSPEED_STAGE2} \
    --task_name ${TASK_NAME} \
    --run_name $(date +%Y%m%d-%H%M%S) \
    --model_name_or_path ${BASE_MODEL_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs ${NUM_TRAIN_EPOCHS} \
    --num_early_stopping_train_epochs ${NUM_EARLY_STOPPING_TRAIN_EPOCHS} \
    --data_seed 10042 \
    --save_strategy ${SAVE_STRATEGY} \
    ${SAVE_TOTAL_LIMIT} \
    --eval_strategy steps \
    --eval_dataset_size ${EVAL_DATASET_SIZE} \
    --save_steps ${SAVE_STEPS} \
    --eval_steps ${EVAL_STEPS} \
    --warmup_steps ${WARMUP_STEPS} \
    --max_train_samples ${MAX_TRAIN_SAMPLES} \
    --max_eval_samples ${MAX_EVAL_SAMPLES} \
    --dataloader_num_workers 3 \
    --logging_strategy steps \
    --logging_steps ${LOGGING_STEPS} \
    --report_to all \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --max_memory_MB ${MAX_MEMORY_MB} \
    --bits ${BITS} \
    --lora_r ${LORA_R} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_dropout 0.05 \
    --lora_modules all \
    --double_quant \
    --quant_type nf4 \
    --bf16 \
    --sliding_window ${SLIDING_WINDOW} \
    --rope_theta ${ROPE_THETA} \
    --dataset ${DATASET} \
    --dataset_format ${DATASET_FORMAT} \
    --prompt_type ${PROMPT_TYPE} \
    --max_new_tokens ${MODEL_MAX_LENGTH} \
    --model_max_len ${MODEL_MAX_LENGTH} \
    --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --per_device_eval_batch_size 1 \
    --learning_rate ${LEARNING_RATE} \
    --lr_scheduler_type ${LR_SCHEDULER_TYPE} \
    --weight_decay 0.0 \
    --seed 10042 \
    --optim ${OPTIM} \
    --gradient_checkpointing True \
    --group_by_length ${GROUP_BY_LENGTH} \
    --ddp_find_unused_parameters False \
    --force_remove_overlength_samples False \
    --flash_attention True \
    --rerope False \
    ${ADD_REASONING_TOKENS} \
    ${NEFTUNE} \
    ${DEEPSEED} \
    ${MISC_PARAMS}