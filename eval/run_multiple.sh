#!/bin/bash
# Usage: run_multiple.sh model_path

SCRIPT_PATH=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
PARENT_PATH=$(cd "${SCRIPT_PATH}/.." ; pwd)

TEST_MODEL_PATH=$1
TASK_NAME=$(basename ${TEST_MODEL_PATH})

COMPLETION_LIMIT=$2
if [ -z ${COMPLETION_LIMIT} ]; then
	COMPLETION_LIMIT=1
fi

echo "Model Path: ${TEST_MODEL_PATH}"
echo "Task Name: ${TASK_NAME}"

# MULTIPL_E_RESULTS_DIR=eval_results/multipl_e/${TASK_NAME}
MULTIPL_E_RESULTS_DIR=eval_results/multipl_e/${TASK_NAME}/$(COMPLETION_LIMIT)-completions/multiple

python ${SCRIPT_PATH}/multiple.py \
		eval \
		--results_dir ${MULTIPL_E_RESULTS_DIR} \
    && \
python ${SCRIPT_PATH}/multiple.py \
		results \
		--results_dir ${MULTIPL_E_RESULTS_DIR}