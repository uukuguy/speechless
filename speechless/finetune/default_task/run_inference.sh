#!/usr/bin/env bash
SCRIPT_ROOT=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
PARENT_PATH=$(cd "${SCRIPT_PATH}/.." ; pwd)

source ${SCRIPT_ROOT}/task.env

TEST_FILE="test_data.jsonl"
OUTPUT_FILE="inference_results.jsonl"

python ${SCRIPT_ROOT}/inference.py \
    --model_path ${TEST_MODEL_PATH} \
    --test_file ${TEST_FILE} \
    --output_file ${OUTPUT_FILE} \
    --tensor_parallel_size ${TENSOR_PARALLEL_SIZE} \
    --batch_size $((${TENSOR_PARALLEL_SIZE} * 4)) \
    --temperature 0.1 \
    --max_tokens 2048 

