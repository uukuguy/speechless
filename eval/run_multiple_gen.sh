#!/bin/bash
# Usage: run_multiple_gen.sh model_path

SCRIPT_PATH=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
PARENT_PATH=$(cd "${SCRIPT_PATH}/.." ; pwd)

TEST_MODEL_PATH=$1
TASK_NAME=$(basename ${TEST_MODEL_PATH})

COMPLETION_LIMIT=$2
if [ -z ${COMPLETION_LIMIT} ]; then
	COMPLETION_LIMIT=1
fi

PARALLEL_THREADS=$3
if [ -z ${PARALLEL_THREADS} ]; then
	PARALLEL_THREADS=5
fi

echo "Model Path: ${TEST_MODEL_PATH}"
echo "Task Name: ${TASK_NAME}"

OUTPUT_DIR=eval_results/multipl_e
MULTIPL_E_RESULTS_DIR=${OUTPUT_DIR}/${TASK_NAME}/${COMPLETION_LIMIT}-completions

# python ${SCRIPT_PATH}/multiple.py \
# 		generate \
# 		--name ${TEST_MODEL_PATH} \
# 		--root-dataset humaneval \
# 		--temperature 0.2 \
# 		--batch-size 20 \
# 		--completion-limit 10 \
# 		--output-dir-prefix ${MULTIPL_E_RESULTS_DIR} \
# 		--langs py java js cpp rs go sh jl \

	# --langs py java js cpp rs go sh jl ts \

python eval/multiple_gen.py \
	--do_generate \
	--output_dir ${MULTIPL_E_RESULTS_DIR} \
	-m ${TASK_NAME}  \
	--langs py js java cpp swift php d jl lua r rkt rs go sh \
	--sampling_method normal \
	--temperature 0.2 \
	--top_p 0.95 \
	--parallel_threads ${PARALLEL_THREADS} \
	--timeout 30 \
	--completion_limit ${COMPLETION_LIMIT} && \
python eval/multiple_gen.py \
	--do_convert \
	--output_dir ${MULTIPL_E_RESULTS_DIR} 
