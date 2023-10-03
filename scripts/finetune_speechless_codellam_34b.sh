#!/bin/bash
export MODELS_ROOT_DIR=/opt/local/llm_models/huggingface.co
export TASK_NAME=speechless-codellama-34b-2.0
# export WANDB_PROJECT=${TASK_NAME}
export OUTPUT_DIR=./outputs
# export DATASET=/opt/local/datasets/jondurbin/airoboros-2.2/instructions-clean.jsonl 
export DATASET=/opt/local/datasets/Speechless/airoboros-orca-platypus-instructions.jsonl
export TORCH_DISTRIBUTED_DEBUG=DETAIL 

# export ENABLE_FLASH_ATTENTION=True
# export ENABLE_REROPE=False

# torchrun --nnodes=1 --nproc_per_node=2 \
#     --deepspeed config/deepspeed-13b-stage2.json \
# python

torchrun --nnodes=1 --nproc_per_node=2 \
    ./finetune.py \
    --task_name ${TASK_NAME} \
    --run_name $(date +%Y%m%d-%H%M%S) \
    --model_name_or_path ${MODELS_ROOT_DIR}/Phind/Phind-CodeLlama-34B-v2 \
    --output_dir ${OUTPUT_DIR}/${TASK_NAME} \
    --num_train_epochs 3 \
    --data_seed 10042 \
    --save_strategy steps \
    --save_total_limit 1 \
    --evaluation_strategy steps \
    --eval_dataset_size 0.005 \
    --save_steps 100 \
    --eval_steps 100 \
    --warmup_steps 40 \
    --max_eval_samples 200 \
    --max_new_tokens 8192 \
    --dataloader_num_workers 3 \
    --logging_strategy steps \
    --logging_steps 1 \
    --report_to tensorboard \
    --remove_unused_columns False \
    --do_train \
    --bits 4 \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_modules all \
    --double_quant \
    --quant_type nf4 \
    --bf16 \
    --dataset ${DATASET} \
    --dataset_format airoboros \
    --model_max_len 8192 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 32 \
    --per_device_eval_batch_size 1 \
    --learning_rate 2e-4 \
    --lr_scheduler_type cosine \
    --weight_decay 0.0 \
    --seed 10042 \
    --optim paged_adamw_8bit \
    --gradient_checkpointing True \
    --group_by_length False \
    --ddp_find_unused_parameters False \
    --force_remove_overlength_samples True \
    --flash_attention True \
    --rerope False
