#!/bin/bash
SCRIPT_PATH=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
PARENT_PATH=$(cd "${SCRIPT_PATH}/.." ; pwd)

source ${SCRIPT_PATH}/task.env

PYTHONPATH=${PWD}/../.. \
python ../../infer.py \
    --model_name ${TASK_NAME} \
    --questions_file /opt/local/datasets/Speechless/chip2023_test_b.json \
    --output_file chip2023_answers.json \
    --parallel 8 \
    --max_tokens 4096