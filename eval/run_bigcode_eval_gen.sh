#!/bin/bash
# Usage: run_bigcode_eval.sh model_path

SCRIPT_PATH=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
PARENT_PATH=$(cd "${SCRIPT_PATH}/.." ; pwd)

TEST_MODEL_PATH=$1
TASK_NAME=$(basename ${TEST_MODEL_PATH})

echo "Model Path: ${TEST_MODEL_PATH}"
echo "Task Name: ${TASK_NAME}"

BIGCODE_TASKS="humaneval,mbpp,multiple-py,multiple-java,multiple-js,multiple-cpp,multiple-rs,multiple-go,multiple-sh,multiple-jl"

# # Don't run generation but benchmark groundtruth (useful for debugging)
# BIGCODE_CHECK_REFERENCES="--check_references"

BITS="--load_in_8bit" 
MAX_LENGTH_GENERATION=2048
TEMPERATURE=0.2
BITS="--load_in_8bit"
PRECISION=bf16
LIMIT=100
N_SAMPLES=1
BATCH_SIZE=1

accelerate launch \
    --num_processes=2 \
    --num_machines=1 \
    --mixed_precision=${PRECISION} \
    --dynamo_backend=no \
    eval/bigcode_eval.py \
    --model ${TEST_MODEL_PATH} \
    ${BITS} \
    --tasks ${BIGCODE_TASKS} \
    --max_length_generation ${MAX_LENGTH_GENERATION} \
    --temperature ${TEMPERATURE} \
    --limit ${LIMIT} \
    --do_sample \
    --n_samples ${N_SAMPLES} \
    --batch_size ${BATCH_SIZE} \
    --precision ${PRECISION}\
    --trust_remote_code \
    --eval_results_dir eval_results/bigcode_eval/${TASK_NAME} \
    --generation_only \
    --save_generations \
    ${BIGCODE_CHECK_REFERENCES}