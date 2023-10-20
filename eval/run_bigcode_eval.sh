#!/bin/bash
SCRIPT_PATH=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
PARENT_PATH=$(cd "${SCRIPT_PATH}/.." ; pwd)

TEST_MODEL_PATH=$1
TASK_NAME=$(basename ${TEST_MODEL_PATH})

TASK=$2

TASK_RESULTS_DIR=${PWD}/eval_results/bigcode_eval/${TASK_NAME}
TASK_GENERATIONS_FILE=bigcode_${TASK}_generations.json
TASK_GENERATIONS_PATH=${TASK_RESULTS_DIR}/${TASK_GENERATIONS_FILE} 
TASK_METRIC_RESULTS_FILE=bigcode_${TASK}_results.json

TEMPERATURE=0.2
N_SAMPLES=1
LIMIT=100

docker run -it --rm \
    -v ${TASK_RESULTS_DIR}:/eval_results:rw \
    evaluation-harness-multiple \
    python3 main.py \
    --model ${TEST_MODEL_PATH} \
    --tasks ${TASK} \
    --load_generations_path /eval_results/${TASK_GENERATIONS_FILE} \
    --metric_output_path /eval_results/${TASK_METRIC_RESULTS_FILE} \
    --allow_code_execution  \
    --limit ${LIMIT} \
    --temperature ${TEMPERATURE} \
    --n_samples ${N_SAMPLES}