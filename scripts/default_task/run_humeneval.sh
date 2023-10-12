#!/bin/bash
# From speechless-code-mistral-7b-v1.0
SCRIPT_PATH=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
PARENT_PATH=$(cd "${SCRIPT_PATH}/.." ; pwd)

source ${SCRIPT_PATH}/task.env

HUMANEVAL_GEN_OUTPUT_FILE=eval_results/human_eval/${TASK_NAME}/humaneval_samples.jsonl

cd ${SCRIPT_PATH}/../.. \
python ./eval/run_humaneval.py \
    ${HUMANEVAL_GEN_OUTPUT_FILE} \
    --problem_file ./eval/datasets/openai_humaneval/HumanEval.jsonl.gz