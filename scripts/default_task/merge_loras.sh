#!/usr/bin/env bash
# From speechless-code-mistral-7b-v1.0
SCRIPT_PATH=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
PARENT_PATH=$(cd "${SCRIPT_PATH}/.." ; pwd)

source ${SCRIPT_PATH}/task.env

# MODELS_ROOT_DIR=/opt/local/llm_models/huggingface.co

# MODEL_BASENAME=$(basename ${PWD})
#CKPT_STEPS=2871
#CKPT_STEPS=5742
CKPT_STEPS=latest

# BASE_MODEL_PATH=${MODELS_ROOT_DIR}/llm_agents/tora-code-7b-v1.0
# BASE_MODEL_PATH=${MODELS_ROOT_DIR}/mistralai/Mistral-7B-v0.1
# BASE_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-coding-7b-16k-tora-2714-steps
# BASE_MODEL_PATH=${MODELS_ROOT_DIR}/spechlessai/speechless-mistral-six-in-one-7b
CHECKPOINT_DIR=${PWD}/outputs/checkpoint-${CKPT_STEPS}/adapter_model
TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/${MODEL_BASENAME}-${CKPT_STEPS}-steps

python merge_peft_adapters.py \
    --base_model_name_or_path ${BASE_MODEL_PATH} \
    --lora_model_path ${CHECKPOINT_DIR} \
    --merged_model_name_or_path ${TEST_MODEL_PATH}