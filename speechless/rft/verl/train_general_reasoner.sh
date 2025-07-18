#! /bin/bash

USER_ENV=`whoami`
set -x
export NCCL_DEBUG=DEBUG
export RAY_BACKEND_LOG_LEVEL=debug
export RAY_DEDUP_LOGS=1

export HEAD_IP=$(hostname)
# export RAY_ADDRESS=$(hostname):8265

export SCRIPT_DIR=$(cd $(dirname $0); pwd)
export WORKING_DIR=${SCRIPT_DIR}
export TASK_NAME=$(basename ${WORKING_DIR})
export CURRENT_TIME=$(date +%Y%m%d_%H%M%S)

export PROJECT_NAME=${TASK_NAME} # to be replaced
# export WANDB_API_KEY= # to be replaced
export WANDB_OFFICIAL=1
# export VLLM_ATTENTION_BACKEND=XFORMERS
export HDFS_DATA_PATH=${SCRIPT_DIR}/data # to be replaced
export HDFS_MODEL_PATH=/opt/local/llm_models/huggingface.co # to be replaced
export HDFS_CHECKPOINT_PATH=${SCRIPT_DIR}/outputs # to be replaced
export HDFS_LOG_PATH=${SCRIPT_DIR}/logs # to be replaced
if [ ! -d "$HDFS_LOG_PATH" ]; then
    mkdir -p $HDFS_LOG_PATH
fi

if [ -z "$RUN_NAME" ]; then
    RUN_NAME=${CURRENT_TIME}
fi
LOG_FILE_PATH="$HDFS_LOG_PATH/$RUN_NAME.log"

# Calculate the number of GPUs based on the CUDA_VISIBLE_DEVICES variable. Use all the GPUs if not set.
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
  export NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
  # Automatically set CUDA_VISIBLE_DEVICES to all available GPUs
  export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS - 1)))
else
  IFS=',' read -r -a gpu_array <<< "$CUDA_VISIBLE_DEVICES"
  export NUM_GPUS=${#gpu_array[@]}
fi

# MODEL_NAME=Qwen/Qwen3-4B
MODEL_NAME=speechlessai/DIFC2025-round_a-SFT-Qwen3-4B
DATASET_NAME=difc2025-round-a
MAX_PROMPT_LENGTH=8192
MAX_RESPONSE_LENGTH=64
# MAX_PROMPT_LENGTH=512
# MAX_RESPONSE_LENGTH=1024

SAVE_FREQ=500
TEST_FREQ=100

TOTAL_EPOCHS=10
# 0.0001
KL_LOSS_COEF=0.0
# 0.001
KL_COEF=0.0
# 5e-7
LEARNING_RATE=2e-6
TENSOR_PARALLEL_SIZE=4
CLIP_RATIO=0.3

# # Qwen3 /think
# TEMPERATURE=0.6
# TOP_P=0.95
# TOP_K=20
# MIN_P=0.0

# Qwen3 /no_think
TEMPERATURE=0.7
TOP_P=0.8
TOP_K=20
MIN_P=0.0

ROLLOUT_N=4
TRAIN_BATCH_SIZE=${ROLLOUT_N}

# MIN_BATCH_SIZE=${ROLLOUT_N}
MIN_BATCH_SIZE=1
PPO_MINI_BATCH_SIZE=${MIN_BATCH_SIZE}
PPO_MICRO_BATCH_SIZE=${MIN_BATCH_SIZE}
LOG_PROB_MICRO_BATCH_SIZE=${MIN_BATCH_SIZE}

ROLLOUT_GPU_MEMORY_UTIL=0.5
ACTOR_OPTIMIZER_OFFLOAD=False
ACTOR_PARAMETER_OFFLOAD=False

CLIP_RATIO=0.3
ENTROPY_COEFFIENT=0.001
KL_LOSS_TYPE="low_var_kl"

VERIFIER_NAME=TIGER-Lab/general-verifier

echo "Arguments received: $@"

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
  echo "Processing: $1"
  case "$1" in
    --train_batch_size) TRAIN_BATCH_SIZE="$2"; shift 2 ;;
    --max_prompt_length) MAX_PROMPT_LENGTH="$2"; shift 2 ;;
    --max_response_length) MAX_RESPONSE_LENGTH="$2"; shift 2 ;;
    --learning_rate) LEARNING_RATE="$2"; shift 2 ;;
    --ppo_mini_batch_size) PPO_MINI_BATCH_SIZE="$2"; shift 2 ;;
    --ppo_micro_batch_size) PPO_MICRO_BATCH_SIZE="$2"; shift 2 ;;
    --kl_loss_coef) KL_LOSS_COEF="$2"; shift 2 ;;
    --entropy_coeffient) ENTROPY_COEFFIENT="$2"; shift 2 ;;
    --clip_ratio) CLIP_RATIO="$2"; shift 2 ;;
    --kl_loss_type) KL_LOSS_TYPE="$2"; shift 2 ;;
    --temperature) TEMPERATURE="$2"; shift 2 ;;
    --log_prob_micro_batch_size) LOG_PROB_MICRO_BATCH_SIZE="$2"; shift 2 ;;
    --rollout_n) ROLLOUT_N="$2"; shift 2 ;;
    --rollout_gpu_memory_util) ROLLOUT_GPU_MEMORY_UTIL="$2"; shift 2 ;;
    --kl_coef) KL_COEF="$2"; shift 2 ;;
    --actor_optimizer_offload) ACTOR_OPTIMIZER_OFFLOAD="$2"; shift 2 ;;
    --actor_parameter_offload) ACTOR_PARAMETER_OFFLOAD="$2"; shift 2 ;;
    --total_epochs) TOTAL_EPOCHS="$2"; shift 2 ;;
    --dataset_name) DATASET_NAME="$2"; shift 2 ;;
    --model_name) MODEL_NAME="$2"; shift 2 ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo "Training with the following parameters:"
echo "Train Batch Size: $TRAIN_BATCH_SIZE"
echo "Max Prompt Length: $MAX_PROMPT_LENGTH" 
echo "Max Response Length: $MAX_RESPONSE_LENGTH" 
echo "Learning Rate: $LEARNING_RATE" 
echo "PPO Mini Batch Size: $PPO_MINI_BATCH_SIZE" 
echo "PPO Micro Batch Size: $PPO_MICRO_BATCH_SIZE" 
echo "KL Loss Coefficient: $KL_LOSS_COEF" 
echo "KL Loss Type: $KL_LOSS_TYPE" 
echo "Temperature: $TEMPERATURE" 
echo "Top P: $TOP_P" 
echo "Top K: $TOP_K" 
echo "Min P: $MIN_P" 
echo "Rollout N: $ROLLOUT_N" 
echo "KL Coefficient: $KL_COEF" 
echo "Total Epochs: $TOTAL_EPOCHS"
echo "Dataset Name: $DATASET_NAME"
echo "Model Name: $MODEL_NAME"
echo "LOG FILE PATH: $LOG_FILE_PATH"

max_num_batched_tokens=$(expr $MAX_PROMPT_LENGTH + $MAX_RESPONSE_LENGTH + 1000)
echo -e "Training with the following parameters:\nTrain Batch Size: $TRAIN_BATCH_SIZE\nVal Batch Size: Max Prompt Length: $MAX_PROMPT_LENGTH\nMax Response Length: $MAX_RESPONSE_LENGTH\nLearning Rate: $LEARNING_RATE\nPPO Mini Batch Size: $PPO_MINI_BATCH_SIZE\nPPO Micro Batch Size: $PPO_MICRO_BATCH_SIZE\nKL Loss Coefficient: $KL_LOSS_COEF\nKL Loss Type: $KL_LOSS_TYPE\nTemperature: $TEMPERATURE\nRollout N: $ROLLOUT_N\nKL Coefficient: $KL_COEF\nTotal Epochs: $TOTAL_EPOCHS\nDataset Name: $DATASET_NAME\nModel Name: $MODEL_NAME"



export PYTHONPATH=${SPEECHLESS_ROOT:-${HOME}/sandbox/LLM/speechless.ai/speechless} 

    # --entrypoint-num-cpus=1 \

    # -- python -m speechless.reasoning.general_reasoner \


# HYDRA_FULL_ERROR=1 ray job submit --address=http://${HEAD_IP}:8265 --working-dir . \
#     --runtime-env-json='{
#          "working_dir": "'${WORKING_DIR}'",
#          "env_vars": {
#             "http_proxy": "",
#             "https_proxy": "",
#             "PYTHONPATH": "'${PYTHONPATH}'",
#             "CUDA_VISIBLE_DEVICES": "'${CUDA_VISIBLE_DEVICES}'"
#          }
#      }' \
#     -- python -m verl.trainer.main_ppo \

if ! python -c "import rouge_chinese" &> /dev/null; then
    pip install rouge-chinese
fi
if ! python -c "import jieba" &> /dev/null; then
    pip install jieba
fi

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    custom_reward_function.path=./compute_score.py \
    reward_model.enable=False \
    reward_model.model.path=$HDFS_MODEL_PATH/$VERIFIER_NAME \
    reward_model.strategy=verifier \
    reward_model.reward_manager=naive \
    reward_model.micro_batch_size=0 \
    data.train_files=[$HDFS_DATA_PATH/$DATASET_NAME/train.parquet] \
    data.val_files=[$HDFS_DATA_PATH/$DATASET_NAME/val.parquet] \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    actor_rollout_ref.model.path=$HDFS_MODEL_PATH/$MODEL_NAME \
    actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
    actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFFIENT \
    actor_rollout_ref.actor.clip_ratio=$CLIP_RATIO \
    actor_rollout_ref.actor.kl_loss_type=$KL_LOSS_TYPE \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=$ACTOR_PARAMETER_OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$ACTOR_OPTIMIZER_OFFLOAD \
    actor_rollout_ref.rollout.temperature=$TEMPERATURE \
    actor_rollout_ref.rollout.top_p=$TOP_P \
    actor_rollout_ref.rollout.top_k=$TOP_K \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=$LOG_PROB_MICRO_BATCH_SIZE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TENSOR_PARALLEL_SIZE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTIL \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.ref.log_prob_micro_batch_size=$LOG_PROB_MICRO_BATCH_SIZE \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    algorithm.kl_ctrl.kl_coef=$KL_COEF \
    critic.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$RUN_NAME \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.nnodes=1 \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.default_local_dir=$HDFS_CHECKPOINT_PATH/$RUN_NAME \
    trainer.total_epochs=$TOTAL_EPOCHS 2>&1 | tee -a $LOG_FILE_PATH
