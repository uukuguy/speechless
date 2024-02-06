#!/usr/bin/env bash
# From speechless-code-mistral-7b-v1.0
SCRIPT_ROOT=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
PARENT_PATH=$(cd "${SCRIPT_PATH}/.." ; pwd)

source ${SCRIPT_ROOT}/task.env

CHECKPOINT_DIR=${TASK_CHECKPOINT_DIR}/checkpoint-2122/adapter_model

python ${SCRIPT_ROOT}/merge_peft_adapters.py \
    --base_model_name_or_path ${BASE_MODEL_PATH} \
    --lora_model_path ${CHECKPOINT_DIR} \
    --merged_model_name_or_path ${TEST_MODEL_PATH} \
