#!/bin/bash
"""
Reference configuration for RL training with the Qwen2.5-72B model using 8 x 80GB GPUs (increase lora_rank if needed):

data.train_batch_size=64 \
actor_rollout_ref.model.use_shm=True \
actor_rollout_ref.model.lora_rank=32 \
actor_rollout_ref.model.lora_alpha=32 \
actor_rollout_ref.model.target_modules=all-linear \
actor_rollout_ref.actor.optim.lr=3e-5 \
actor_rollout_ref.actor.fsdp_config.fsdp_size=8 \
actor_rollout_ref.actor.fsdp_config.param_offload=True \
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
actor_rollout_ref.rollout.name=vllm \
actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
actor_rollout_ref.rollout.n=5 \
actor_rollout_ref.rollout.max_num_seqs=64 \
actor_rollout_ref.rollout.max_model_len=1536 \
actor_rollout_ref.rollout.max_num_batched_tokens=1536 \
actor_rollout_ref.rollout.load_format=safetensors \
actor_rollout_ref.rollout.layered_summon=True \
actor_rollout_ref.ref.fsdp_config.param_offload=True \
actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
"""

set -x # Enable debugging
set -e # Stop on error
set -u # Stop if an undefined variable is referenced

# export VLLM_ATTENTION_BACKEND=XFORMERS

# ----- Project specific configurations -----
export WANDB_PROJECT=$(basename $(pwd))
export RUN_NAME=$(date +%Y%m%d-%H%M%S)
mkdir -p logs
LOG_FILE=logs/verl_grpo-${RUN_NAME}.log

# ----- Reward function configurations -----
REWARD_SCORE_FILE=./reward_score.py
REWARD_FUNCTION_NAME=compute_score

# ----- Model configurations -----
# FIXME
REF_MODEL_PATH=/opt/local/llm_models/huggingface.co/Qwen/Qwen3-4B
TENSOR_MODEL_PARALLEL_SIZE=1

# ----- Data configurations -----
TRAIN_FILES=./data/natural_reasoning_finance/train.parquet
VAL_FILES=./data/natural_reasoning_finance/test.parquet
DATA_SHUFFLE=False

TRAIN_BATCH_SIZE=64
MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=2048
NUM_TRAIN_EPOCHS=1

# ----- GPU configurations -----
NUM_GPUS=8
NNODES=1

# ----- Rollout configurations -----
ROLLOUT_N=8
# FIXME
ROLLOUT_GPU_MEMORY_UTILIZATION=0.8 # A value between 0.5 and 0.7 often strikes a good balance between high throughput and avoiding OOM. 4B: 0.8, 8B: 0.7, 32B: 0.6, 72B: 0.4
ROLLOUT_CALCULATE_LOG_PROBS=True # ++actor_rollout_ref.rollout.calculate_log_probs=${ROLLOUT_CALCULATE_LOG_PROBS} verl v0.4.1


# ----- Training configurations -----
# FIXME
LR_ACTOR=3e-6
ENTROPY_COEFF=0.001 # FIXME for length control
PPO_MINI_BATCH_SIZE=512

# ----- LoRA configurations -----
# FIXME
# https://verl.readthedocs.io/en/latest/advance/ppo_lora.html
LORA_RANK=64
LORA_ALPHA=64

MODEL_USE_SHM=True # actor_rollout_ref.model.use_shm=True: preload the model into /dev/shm to improve model loading speed.
ROLLOUT_LAYERED_SUMMON=True # actor_rollout_ref.rollout.layered_summon=True: this enables the actor-model to gather the FSDP shards per layers when synchronizing the LoRA Adapter to vLLM, thereby reducing GPU peak memory. Recommended if the model is very large (70B+) or the GPU memory is limited (< 48GB)


# ----- Saving checkpoints configurations -----
SAVE_DIR=./checkpoints/e3_4b/
SAVE_FREQ=5
TEST_FREQ=1

# Warning: remove_previous_ckpt_in_save is deprecated," + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead
MAX_CKPT_TO_KEEP=1
    # ++trainer.max_actor_ckpt_to_keep=${MAX_CKPT_TO_KEEP} \
    # ++trainer.max_critic_ckpt_to_keep=${MAX_CKPT_TO_KEEP} \

# ----- FSDP configurations -----
# FIXME
FSDP_SIZE=${NUM_GPUS}
FSDP_PARAM_OFFLOAD=True
FSDP_OPTIMIZER_OFFLOAD=False
FSDP_GRAD_OFFLOAD=False
    # actor_rollout_ref.actor.fsdp_config.param_offload=${FSDP_PARAM_OFFLOAD} \
    # actor_rollout_ref.actor.fsdp_config.optimizer_offload=${FSDP_OPTIMIZER_OFFLOAD} \
    # actor_rollout_ref.actor.fsdp_config.grad_offload=${FSDP_GRAD_OFFLOAD} \

ULYSSES_SEQUENCE_PARALLEL_SIZE=1
    # actor_rollout_ref.actor.ulysses_sequence_parallel_size=${ULYSSES_SEQUENCE_PARALLEL_SIZE} \

# PPO_MAX_TOKEN_LEN=25024
# ROLLOUT_TEMPERATURE=1.0
# ROLLOUT_VAL_TEMPERATURE=0.6

    # actor_rollout_ref.actor.use_dynamic_bsz=True \
    # actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MINI_BATCH_SIZE} \
    # actor_rollout_ref.rollout.temperature=${ROLLOUT_TEMPERATURE} \
    # +actor_rollout_ref.rollout.val_temperature=${ROLLOUT_VAL_TEMPERATURE} \
    # algorithm.kl_ctrl.kl_coef=0.001 \

# -------- E3 --------
enable_temperature_scheduler=False
enable_annealing=False

n_rollout_update=2
n_rollout_max=12
n_rollout_min=4

# response length control
# add a length-based bias to the rewards of correct rollouts for the bottom (p%) of questions ranked by difficulty (simple questions)
PENALTY_COEFF=0.001
PENALTY_THRESHOLD=0.1

# Enable activation offloading: This often works together with gradient checkpointing to get larger micro-batch sizes and it’s only available in FSDP backend now.
ENABLE_ACTIVATION_OFFLOAD=False

    # ++actor_rollout_ref.model.enable_activation_offload=${ENABLE_ACTIVATION_OFFLOAD} \
    # ++critic.model.enable_activation_offload=${ENABLE_ACTIVATION_OFFLOAD} \

VAL_BEFORE_TRAIN=True

export TENSORBOARD_DIR=$SAVE_DIR/tensorboard
export HYDRA_FULL_ERROR=1

# FSDP2 and training optimizations (verl v0.4.0)
# FSDP2 is recommended to replace FSDP1, providing better throughput and memory usage, and is composable with other features (e.g. torch.compile):

    # ++actor_rollout_ref.ref.strategy=fsdp2 \
    # ++actor_rollout_ref.actor.strategy=fsdp2 \
    # ++critic.strategy=fsdp2 \
    # ++reward_model.strategy=fsdp2 \
    # ++actor_rollout_ref.actor.offload_policy=True \

# Furthermore, FSDP2 cpu offloading is compatible with gradient accumulation. You can turn it on to save memory with actor_rollout_ref.
# actor.offload_policy=True.

# Fused cross entropy kernel to drastically reduce peak memory. (verl v0.4.0)

    # ++actor_rollout_ref.model.use_fused_kernels=True

# PYTHONPATH=$PWD/verl \

# python3 -m e3.main_e3 \

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
    data.shuffle=${DATA_SHUFFLE} \
    actor_rollout_ref.model.path=${REF_MODEL_PATH} \
    ++actor_rollout_ref.model.use_shm=${MODEL_USE_SHM} \
    ++actor_rollout_ref.rollout.layered_summon=${ROLLOUT_LAYERED_SUMMON} \
    ++actor_rollout_ref.model.lora_rank=${LORA_RANK} \
    ++actor_rollout_ref.model.lora_alpha=${LORA_ALPHA} \
    actor_rollout_ref.actor.optim.lr=${LR_ACTOR} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${FSDP_SIZE} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${FSDP_PARAM_OFFLOAD} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${FSDP_PARAM_OFFLOAD} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${FSDP_OPTIMIZER_OFFLOAD} \
    ++actor_rollout_ref.actor.fsdp_config.grad_offload=${FSDP_GRAD_OFFLOAD} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${ULYSSES_SEQUENCE_PARALLEL_SIZE} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${TENSOR_MODEL_PARALLEL_SIZE} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION} \
    ++actor_rollout_ref.rollout.n_low=$n_rollout_min \
    ++actor_rollout_ref.rollout.n_high=$n_rollout_max \
    ++actor_rollout_ref.rollout.n_update=$n_rollout_update \
    actor_rollout_ref.rollout.temperature=1 \
    ++actor_rollout_ref.rollout.enable_temperature_scheduler=$enable_temperature_scheduler \
    ++actor_rollout_ref.rollout.enable_annealing=$enable_annealing \
    ++actor_rollout_ref.rollout.max_steps=480 \
    ++actor_rollout_ref.rollout.calculate_log_probs=${ROLLOUT_CALCULATE_LOG_PROBS} \
    ++actor_rollout_ref.model.use_fused_kernels=True \
    ++actor_rollout_ref.model.enable_activation_offload=${ENABLE_ACTIVATION_OFFLOAD} \
    ++critic.model.enable_activation_offload=${ENABLE_ACTIVATION_OFFLOAD} \
    algorithm.use_kl_in_reward=False \
    ++trainer.max_actor_ckpt_to_keep=${MAX_CKPT_TO_KEEP} \
    ++trainer.max_critic_ckpt_to_keep=${MAX_CKPT_TO_KEEP} \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name=${WANDB_PROJECT} \
    trainer.experiment_name=${RUN_NAME} \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.nnodes=${NNODES} \
    trainer.default_local_dir=$SAVE_DIR \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.val_before_train=${VAL_BEFORE_TRAIN} \
    ++algorithm.penalty_coeff=${PENALTY_COEFF} \
    ++algorithm.penalty_threshold=${PENALTY_THRESHOLD} \
    trainer.total_epochs=${NUM_TRAIN_EPOCHS} $@ 2>&1 | tee ${LOG_FILE}

