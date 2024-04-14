#!/bin/bash
SCRIPT_PATH=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
PARENT_PATH=$(cd "${SCRIPT_PATH}/.." ; pwd)

source ${SCRIPT_PATH}/task.env

# python -m torch.distributed.run \
#     --nproc_per_node $NPROC_PER_NODE \
#     --nnodes $NNODES \
#     --node_rank $RANK \
#     --master_addr $MASTER_ADDR \
#     --master_port $MASTER_PORT \

# deepspeed --num_gpus 4 \

    # --max_samples 3000 \
    # --val_size 0.1 \
    # --plot_loss \
    # --eval_dataset_size ${EVAL_DATASET_SIZE} \
    # --overwrite_cache \
    # --overwrite_output_dir \
    # --quantization_bit 4 \

PYTHONPATH=${SPEECHLESS_ROOT} \
torchrun --nnodes=1 --nproc_per_node=${NUM_GPUS} \
    --master_port 29511 \
    -m speechless.finetune.pretrain \
    --stage sft \
    --do_train \
    --model_name_or_path ${BASE_MODEL_PATH} \
    --dataset ${DATASET} \
    --dataset_dir ../../data \
    --template default \
    --finetuning_type lora \
    --lora_target all \
    --lora_r ${LORA_R} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_dropout 0.05 \
    --output_dir ${OUTPUT_DIR}/pretrain \
    --cutoff_len ${MODEL_MAX_LENGTH} \
    --preprocessing_num_workers 16 \
    --report_to tensorboard \
    --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --warmup_steps ${WARMUP_STEPS} \
    --save_strategy ${SAVE_STRATEGY} \
    ${SAVE_TOTAL_LIMIT} \
    --save_steps ${SAVE_STEPS} \
    --eval_steps ${EVAL_STEPS} \
    --evaluation_strategy steps \
    --learning_rate ${LEARNING_RATE} \
    --num_train_epochs ${NUM_TRAIN_EPOCHS} \
    --max_samples ${MAX_TRAIN_SAMPLES} \
    --val_size ${EVAL_DATASET_SIZE} \
    --ddp_timeout 180000000 \
    --flash_attn \
    --bf16 \
    ${DEEPSEED} \
