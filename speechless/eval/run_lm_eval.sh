#!/bin/bash
# Usage: run_lm_eval.sh model_path

SCRIPT_PATH=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
PARENT_PATH=$(cd "${SCRIPT_PATH}/.." ; pwd)

TEST_MODEL_PATH=$1
TASK_NAME=$(basename ${TEST_MODEL_PATH})

echo "Model Path: ${TEST_MODEL_PATH}"
echo "Task Name: ${TASK_NAME}"

python ${SCRIPT_PATH}/lm_eval.py \
    --model hf-causal \
    --model_args pretrained=${TEST_MODEL_PATH} \
    --task \
        "arc_challenge|25|100" \
        "hellaswag|10|100" \
        "hendrycksTest-*|5|100" \
        "truthfulqa_mc|0|100" \
        "winogrande|5|100" \
        "gsm8k|5|100" \
        "drop|3|100" \
    --batch_size 4 \
    --no_cache \
    --write_out \
    --output_path eval_results/lm_eval/${TASK_NAME} 
