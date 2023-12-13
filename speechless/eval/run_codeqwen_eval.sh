#!/bin/bash
SCRIPT_PATH=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
PARENT_PATH=$(cd "${SCRIPT_PATH}/.." ; pwd)

TEST_MODEL_PATH=$1
TASK_NAME=$(basename ${TEST_MODEL_PATH})

# TASK=$2
# BIGCODE_TASKS="humaneval,multiple-java,multiple-js,multiple-cpp,multiple-rs,multiple-go,multiple-sh,multiple-jl,multiple-swift,multiple-php,multiple-d,multiple-lua,multiple-r,multiple-rkt"

# BIGCODE_TASKS="humaneval multiple-java multiple-js multiple-cpp multiple-rs multiple-jl multiple-swift multiple-php multiple-d multiple-lua multiple-r multiple-rkt"

CODEQWEN_TASKS="humanevalsynthesize-python humanevalsynthesize-java humanevalsynthesize-js humanevalsynthesize-cpp humanevalsynthesize-go humanevalsynthesize-rust humanevalfixtests-python humanevalfixtests-java humanevalfixtests-js humanevalfixtests-cpp humanevalfixtests-go humanevalfixtests-rust mbpp"


TEMPERATURE=0.2
N_SAMPLES=1
# LIMIT=100
# --limit ${LIMIT} \

for TASK in ${CODEQWEN_TASKS}; do
    echo "Running ${task}"
    TASK_RESULTS_DIR=${PWD}/eval_results/bigcode_eval/${TASK_NAME}
    TASK_GENERATIONS_FILE=generations_${TASK}_${TASK_NAME}.json
    TASK_GENERATIONS_PATH=${TASK_RESULTS_DIR}/${TASK_GENERATIONS_FILE} 
    TASK_METRIC_RESULTS_FILE=generations_${TASK}_${TASK_NAME}_results.json
    docker run -it --rm \
        -v ${TASK_RESULTS_DIR}:/eval_results:rw \
        evaluation-harness-multiple \
        python3 main.py \
            --model ${TEST_MODEL_PATH} \
            --tasks ${TASK} \
            --load_generations_path /eval_results/${TASK_GENERATIONS_FILE} \
            --metric_output_path /eval_results/${TASK_METRIC_RESULTS_FILE} \
            --allow_code_execution  \
            --temperature ${TEMPERATURE} \
            --n_samples ${N_SAMPLES}
done;
