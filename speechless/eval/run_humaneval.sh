#!/bin/bash
# Usage: run_humaneval.sh model_path

SCRIPT_PATH=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
PARENT_PATH=$(cd "${SCRIPT_PATH}/.." ; pwd)

TEST_MODEL_PATH=$1
TASK_NAME=$(basename ${TEST_MODEL_PATH})

echo "Model Path: ${TEST_MODEL_PATH}"
echo "Task Name: ${TASK_NAME}"

HUMANEVAL_GEN_OUTPUT_FILE=eval_results/human_eval/${TASK_NAME}/humaneval_samples.jsonl

PYTHONPATH=${SPEECHLESS_ROOT} \
python -m speechless.eval.humaneval \
    ${HUMANEVAL_GEN_OUTPUT_FILE} \
    --problem_file ${SCRIPT_PATH}/datasets/openai_humaneval/HumanEval.jsonl.gz