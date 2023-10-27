#!/usr/bin/env bash
# From speechless-code-mistral-7b-v1.0
SCRIPT_PATH=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
PARENT_PATH=$(cd "${SCRIPT_PATH}/.." ; pwd)

#source ${SCRIPT_PATH}/task.env

MODELS_ROOT_DIR=/opt/local/llm_models/huggingface.co
BASE_MODEL_PATH=${MODELS_ROOT_DIR}/mistralai/Mistral-7B-v0.1
CHECKPOINT_DIR=${MODELS_ROOT_DIR}/multi-loras/Mistral-7B-v0.1/speechless-agents-7b-v0.2-32k-mistral/speechless-agents-7b-v0.2-32k-mistral-397-steps-lora
TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-agents-7b-v0.2-32k-mistral

python ${SPEECHLESS_ROOT}/speechless/scripts/merge_peft_adapters.py \
    --base_model_name_or_path ${BASE_MODEL_PATH} \
    --peft_model_path ${CHECKPOINT_DIR} \
    --merged_model_name_or_path ${TEST_MODEL_PATH} 