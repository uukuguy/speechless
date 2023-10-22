#!/usr/bin/env bash
# From speechless-code-mistral-7b-v1.0
SCRIPT_PATH=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
PARENT_PATH=$(cd "${SCRIPT_PATH}/.." ; pwd)

source ${SCRIPT_PATH}/task.env

# python ${SPEECHLESS_ROOT}/speechless/api/server.py \

PYTHONPATH=${SPEECHLESS_ROOT} \
python -m speechless.api.server \
        --model_name_or_path=${TEST_MODEL_PATH} \
        --model_family vllm