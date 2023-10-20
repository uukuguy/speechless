#!/bin/bash
# Usage: run_multiple_gen.sh model_path

SCRIPT_PATH=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
PARENT_PATH=$(cd "${SCRIPT_PATH}/.." ; pwd)

TEST_MODEL_PATH=$1
TASK_NAME=$(basename ${TEST_MODEL_PATH})

echo "Model Path: ${TEST_MODEL_PATH}"
echo "Task Name: ${TASK_NAME}"

OUTPUT_DIR=eval_results/multipl_e
MULTIPL_E_RESULTS_DIR=${OUTPUT_DIR}/${TASK_NAME}

# python ${SCRIPT_PATH}/multiple.py \
# 		generate \
# 		--name ${TEST_MODEL_PATH} \
# 		--root-dataset humaneval \
# 		--temperature 0.2 \
# 		--batch-size 20 \
# 		--completion-limit 10 \
# 		--output-dir-prefix ${MULTIPL_E_RESULTS_DIR} \
# 		--langs py java js cpp rs go sh jl \

python eval/multiple_gen.py \
	--do_generate \
	--output_dir ${MULTIPL_E_RESULTS_DIR} \
	-m ${TASK_NAME}  \
	--langs py java js cpp rs go sh jl \
	--completion_limit 5 && \
python eval/multiple_gen.py \
	--do_convert \
	--output_dir ${MULTIPL_E_RESULTS_DIR} 
