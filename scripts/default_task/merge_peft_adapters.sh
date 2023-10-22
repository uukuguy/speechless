#!/usr/bin/env bash
# From speechless-code-mistral-7b-v1.0
SCRIPT_PATH=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
PARENT_PATH=$(cd "${SCRIPT_PATH}/.." ; pwd)

source ${SCRIPT_PATH}/task.env

CHECKPOINT_DIR=${TASK_CHECKPOINT_DIR}/checkpoint-2122/adapter_model

python ${SPEECHLESS_ROOT}/speechless/scripts/merge_peft_adapters.py \
    --base_model_name_or_path ${BASE_MODEL_PATH} \
    --peft_model_path ${CHECKPOINT_DIR} \
    --merged_model_name_or_path ${TEST_MODEL_PATH} \