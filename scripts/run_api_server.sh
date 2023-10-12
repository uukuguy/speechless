#!/usr/bin/env bash
# From speechless-code-mistral-7b-v1.0
SCRIPT_PATH=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
PARENT_PATH=$(cd "${SCRIPT_PATH}/.." ; pwd)

source ${SCRIPT_PATH}/task.env

PYTHONPATH=${SCRIPT_PATH}/../.. \
python ../api/server.py \
        --model_name_or_path=$(1) \