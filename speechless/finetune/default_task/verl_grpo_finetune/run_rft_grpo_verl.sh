#!/bin/bash

set -x # Enable debugging
set -e # Stop on error
set -u # Stop if an undefined variable is referenced

# ----- Model configurations -----
export REF_MODEL_PATH=Qwen/Qwen2-7B-Instruct

# ----- Data configurations -----
export TRAIN_FILES=$HOME/data/gsm8k/train.parquet
export VAL_FILES=$HOME/data/gsm8k/test.parquet

# ----- Training configurations -----
export NUM_GPUS=4
export NNODES=1

export TRAIN_BATCH_SIZE=1024
export MAX_PROMPT_LENGTH=512
export MAX_RESPONSE_LENGTH=1024

export NUM_TRAIN_EPOCHS=5
export LR=3e-6
export SAVE_FREQ=-1
export TEST_FREQ=5

export PPO_MINI_BATCH_SIZE=256
export TENSOR_MODEL_PARALLEL_SIZE=2

export ROLLOUT_N=5
export ROLLOUT_GPU_MEMORY_UTILIZATION=0.6

export FSDP_PARAM_OFFLOAD=False
export FSDP_GRAD_OFFLOAD=False

# ----- Reward function configurations -----
export REWARD_SCORE_FILE=./reward_score.py
export REWARD_FUNCTION_NAME=compute_score

# ----- Project specific configurations -----
export WANDB_PROJECT=$(basename $(pwd))
export RUN_NAME=$(date +%Y%m%d-%H%M%S)
export LOG_FILE=logs/verl_grpo-${RUN_NAME}.log

export VLLM_ATTENTION_BACKEND=XFORMERS

# export FSDP_OPTIMIZER_OFFLOAD=True
# export PPO_MAX_TOKEN_LEN=25024
# export ULYSSES_SEQUENCE_PARALLEL_SIZE=1
# export ROLLOUT_TEMPERATURE=1.0
# export ROLLOUT_VAL_TEMPERATURE=0.6

    # actor_rollout_ref.actor.use_dynamic_bsz=True \
    # actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MINI_BATCH_SIZE} \
    # actor_rollout_ref.actor.ulysses_sequence_parallel_size=${ULYSSES_SEQUENCE_PARALLEL_SIZE} \
    # actor_rollout_ref.rollout.temperature=${ROLLOUT_TEMPERATURE} \
    # +actor_rollout_ref.rollout.val_temperature=${ROLLOUT_VAL_TEMPERATURE} \
    # actor_rollout_ref.actor.fsdp_config.optimizer_offload=${FSDP_OPTIMIZER_OFFLOAD} \
    # algorithm.kl_ctrl.kl_coef=0.001 \
    # +trainer.val_before_train=True \

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    custom_reward_function.path=${REWARD_SCORE_FILE} \
    custom_reward_function.name=${REWARD_FUNCTION_NAME} \
    data.train_files=${TRAIN_FILES} \
    data.val_files=${VAL_FILES} \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=False \
    actor_rollout_ref.model.path=${REF_MODEL_PATH} \
    actor_rollout_ref.model.use_shm=True \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=40 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=${FSDP_PARAM_OFFLOAD} \
    actor_rollout_ref.actor.fsdp_config.grad_offload=${FSDP_GRAD_OFFLOAD} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${TENSOR_MODEL_PARALLEL_SIZE} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION} \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=40 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb', 'tensorboard'] \
    trainer.project_name=${WANDB_PROJECT} \
    trainer.experiment_name=${RUN_NAME} \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.total_epochs=${NUM_TRAIN_EPOCHS} $@ 2>&1 | tee ${LOG_FILE}
