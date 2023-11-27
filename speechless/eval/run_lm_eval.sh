#!/bin/bash
# Usage: run_lm_eval.sh model_path

SCRIPT_PATH=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
PARENT_PATH=$(cd "${SCRIPT_PATH}/.." ; pwd)

TEST_MODEL_PATH=$1
TASK_NAME=$(basename ${TEST_MODEL_PATH})
LIMIT=0

echo "Model Path: ${TEST_MODEL_PATH}"
echo "Task Name: ${TASK_NAME}"

    # --model hf-causal \

python ${SCRIPT_PATH}/lm_eval.py \
    --model hf-causal-experimental \
    --model_args pretrained=${TEST_MODEL_PATH},use_accelerate=True \
    --task \
        "arc_challenge|25|${LIMIT}" \
        "hellaswag|10|${LIMIT}" \
        "hendrycksTest-*|5|${LIMIT}" \
        "truthfulqa_mc|0|${LIMIT}" \
        "winogrande|5|${LIMIT}" \
        "gsm8k|5|${LIMIT}" \
        "drop|3|${LIMIT}" \
    --batch_size 8 \
    --no_cache \
    --write_out \
    --output_path eval_results/lm_eval/${TASK_NAME} 
