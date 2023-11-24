#!/bin/bash
# From speechless-code-mistral-7b-v1.0
SCRIPT_PATH=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
PARENT_PATH=$(cd "${SCRIPT_PATH}/.." ; pwd)

source ${SCRIPT_PATH}/task.env

HUMANEVAL_GEN_OUTPUT_FILE=eval_results/human_eval/${TASK_NAME}/humaneval_samples.jsonl

bash ${SPEECHLESS_ROOT}/speechless/eval/run_humaneval_gen.sh \
	${TEST_MODEL_PATH} \
	${HUMANEVAL_GEN_OUTPUT_FILE}