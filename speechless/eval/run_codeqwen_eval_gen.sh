#!/bin/bash
# Usage: run_bigcode_eval.sh model_path

SCRIPT_PATH=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
PARENT_PATH=$(cd "${SCRIPT_PATH}/.." ; pwd)

TEST_MODEL_PATH=$1
TASK_NAME=$(basename ${TEST_MODEL_PATH})

echo "Model Path: ${TEST_MODEL_PATH}"
echo "Task Name: ${TASK_NAME}"

# CUSTOM_TASKS="humaneval,mbpp,multiple-py,multiple-java,multiple-js,multiple-cpp,multiple-rs,multiple-go,multiple-sh,multiple-jl"

# langs=(py js java cpp swift php d jl lua r rkt rs)

# BIGCODE_TASKS="humaneval,multiple-java,multiple-js,multiple-cpp,multiple-rs,multiple-go,multiple-sh,multiple-jl,multiple-swift,multiple-php,multiple-d,multiple-lua,multiple-r,multiple-rkt"

# BIGCODE_TASKS="humaneval,multiple-java,multiple-js,multiple-cpp,multiple-rs,multiple-jl,multiple-swift,multiple-php,multiple-d,multiple-lua,multiple-r,multiple-rkt"

CODEQWEN_TASKS="humanevalsynthesize-python,humanevalsynthesize-java,humanevalsynthesize-js,humanevalsynthesize-cpp,humanevalsynthesize-go,humanevalsynthesize-rust,humanevalfixtests-python,humanevalfixtests-java,humanevalfixtests-js,humanevalfixtests-cpp,humanevalfixtests-go,humanevalfixtests-rust,mbpp"


# BIGCODE_TASKS="humaneval,multiple-*"

# # Don't run generation but benchmark groundtruth (useful for debugging)
# BIGCODE_CHECK_REFERENCES="--check_references"

TEMPERATURE=0.2
MAX_LENGTH_GENERATION=2000
N_SAMPLES=1
TOP_P=0.95

# BITS="--load_in_8bit"
PRECISION=bf16
# PRECISION=bf16
# LIMIT=100
# --limit ${LIMIT} \

BATCH_SIZE=1

accelerate launch \
    --main_process_port 29501 \
    --num_processes=2 \
    --num_machines=1 \
    --mixed_precision=${PRECISION} \
    --dynamo_backend=no \
    ${SPEECHLESS_ROOT}/speechless/eval/bigcode_eval.py \
        --model ${TEST_MODEL_PATH} \
        --prompt instruct \
        ${BITS} \
        --tasks ${CODEQWEN_TASKS} \
        --max_length_generation ${MAX_LENGTH_GENERATION} \
        --temperature ${TEMPERATURE} \
        --top_p ${TOP_P} \
        --do_sample False \
        --n_samples ${N_SAMPLES} \
        --batch_size ${BATCH_SIZE} \
        --precision ${PRECISION}\
        --seed 999999999 \
        --trust_remote_code \
        --eval_results_dir eval_results/bigcode_eval/${TASK_NAME} \
        --generation_only \
        --save_generations \
        ${BIGCODE_CHECK_REFERENCES}