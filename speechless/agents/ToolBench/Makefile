MODELS_ROOT_DIR=/opt/local/llm_models/huggingface.co
#TOOLLLAMA_MODEL_PATH=${MODELS_ROOT_DIR}/ToolBench/ToolLLaMA-2-7b-v2

TOOLLLAMA_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-tools-7b-v0.1-32k-mistral-5743-steps
# TOOLLLAMA_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-tools-7b-multi-rounds-32k-tora-train0.15-861-steps
# TOOLLLAMA_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-tools-7b-v0.1-32k-tora-1436-steps
# TOOLLLAMA_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-tools-7b-v0.1-32k-tora-2872-steps
# TOOLLLAMA_MODEL_PATH=${MODELS_ROOT_DIR}/mistralai/Mistral-7B-v0.1
# TOOLLLAMA_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-agents-7b-v0.2-32k-mistral
# TOOLLLAMA_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-agents-7b-v0.1-tora
RETRIEVAL_MODEL_PATH=${MODELS_ROOT_DIR}/ToolBench/ToolBench_IR_bert_based_uncased

TOOLBENCH_DATA_DIR=/opt/local/datasets/toolbench_data
TOOL_ROOT_DIR=${TOOLBENCH_DATA_DIR}/toolenv/tools/
CORPUS_TSV_PATH=${TOOLBENCH_DATA_DIR}/retrieval/G1/corpus.tsv
INPUT_QUERY_FILE=${TOOLBENCH_DATA_DIR}/test_instruction/G1_instruction.json
RETRIEVED_API_NUMS=5
METHOD=DFS_woFilter_w2

EXEC_ARGS=--max_observation_length 1024 --observ_compress_method truncate --method ${METHOD}

RETRIEVAL_ARGS="--corpus_tsv_path ${CORPUS_TSV_PATH} --retrieval_model_path ${RETRIEVAL_MODEL_PATH} --retrieved_api_nums ${RETRIEVED_API_NUMS} "

MODEL_NAME=$(shell basename ${TOOLLLAMA_MODEL_PATH})
OUTPUT_DIR=${PWD}/outputs/${MODEL_NAME}.${METHOD} 

run_speechless:
	PYTHONPATH=${PWD} \
	python toolbench/inference/qa_pipeline.py \
		--tool_root_dir ${TOOL_ROOT_DIR} \
		--backbone_model toolllama \
		--model_path ${TOOLLLAMA_MODEL_PATH} \
		${EXEC_ARGS} \
		--input_query_file ${INPUT_QUERY_FILE} \
		--output_answer_file ${OUTPUT_DIR} \
		--toolbench_key ${TOOLBENCH_KEY}

run_opendomain:
	PYTHONPATH=${PWD} \
	python toolbench/inference/qa_pipeline_open_domain.py \
		--tool_root_dir ${TOOL_ROOT_DIR} \
		--backbone_model toolllama \
		--model_path ${TOOLLLAMA_MODEL_PATH} \
		${RETRIEVAL_ARGS} \
		${EXEC_ARGS} \
		--input_query_file ${INPUT_QUERY_FILE} \
		--output_answer_file ${PWD}/toolllama_dfs_open_domain_inference_result \
		--toolbench_key ${TOOLBENCH_KEY}


run_rapidapi:
	PYTHONPATH=${PWD} \
	python toolbench/inference/toolbench_server.py \
		--tool_root_dir ${TOOL_ROOT_DIR} \
		--backbone_model toolllama \
		--model_path ${TOOLLLAMA_MODEL_PATH} \
		${RETRIEVAL_ARGS} \
		${EXEC_ARGS} \
		--input_query_file ${INPUT_QUERY_FILE} \
		--output_answer_file ${PWD}/toolllama_dfs_open_domain_result \
		--rapidapi_key ${RAPIDAPI_KEY}


run_toolllama:
	PYTHONPATH=${PWD} \
	python toolbench/inference/qa_pipeline.py \
		--tool_root_dir ${TOOL_ROOT_DIR} \
		--backbone_model toolllama \
		--model_path ${TOOLLLAMA_MODEL_PATH} \
		${EXEC_ARGS} \
		--input_query_file ${INPUT_QUERY_FILE} \
		--output_answer_file ${PWD}/toolllama_dfs_inference_result \
		--toolbench_key ${TOOLBENCH_KEY}


run_openai:
	PYTHONPATH=${PWD} \
	python toolbench/inference/qa_pipeline.py \
		--tool_root_dir ${TOOL_ROOT_DIR} \
		--backbone_model chatgpt_function \
		--openai_key ${OPENAI_KEY} \
		${EXEC_ARGS} \
		--input_query_file ${INPUT_QUERY_FILE} \
		--output_answer_file chatgpt_dfs_inference_result \
		--toolbench_key ${TOOLBENCH_KEY}
		

run_davinci:
	PYTHONPATH=${PWD} \
	python toolbench/inference/qa_pipeline.py \
		--tool_root_dir ${TOOL_ROOT_DIR} \
		--backbone_model davinci \
		--openai_key ${OPENAI_KEY} \
		${EXEC_ARGS} \
		--input_query_file ${INPUT_QUERY_FILE} \
		--output_answer_file chatgpt_dfs_inference_result \
		--toolbench_key ${TOOLBENCH_KEY}


eval_prepare:
	cd toolbench/tooleval
	export RAW_ANSWER_PATH=../../data/my_reproduction_data/model_predictions/
	export CONVERTED_ANSWER_PATH=../../data/my_reproduction_data/model_predictions_converted/
	export MODEL_NAME=chatgpt_cot
	export METHOD=CoT
	mkdir ${CONVERTED_ANSWER_PATH}/${MODEL_NAME}
	for test_set in G1_instruction G1_category G1_tool G2_category G2_instruction G3_instruction
	do
		answer_dir=${RAW_ANSWER_PATH}/${MODEL_NAME}/${test_set}
		output_file=${CONVERTED_ANSWER_PATH}/${MODEL_NAME}/${test_set}.json
		python convert_to_answer_format.py\
			--answer_dir ${answer_dir} \
			--method ${METHOD} \
			--output ${output_file}
	done


pass_rate:
	cd toolbench/tooleval 
	export CONVERTED_ANSWER_PATH=../../data/reproduction_data/model_predictions_converted/ 
	export SAVE_PATH=pass_rate_results 
	export CANDIDATE_MODEL=chatgpt_cot 
	python eval_pass_rate.py \
		--converted_answer_path ${CONVERTED_ANSWER_PATH} \
		--save_path ${SAVE_PATH} \
		--reference_model ${CANDIDATE_MODEL} \
		--test_ids ../../data/test_query_ids/ \
		--max_eval_threads 20 \
		--evaluate_times 4

win_rate:
	cd toolbench/tooleval
	export CONVERTED_ANSWER_PATH=../../data/reproduction_data/model_predictions_converted/
	export SAVE_PATH=preference_results
	export PASS_TARE_PATH=pass_rate_results
	export REFERENCE_MODEL=chatgpt_cot
	export CANDIDATE_MODEL=gpt-4-0613_cot
	# export API_POOL_FILE=path/to/your/openai_key_json_file.json
	python eval_preference.py \
		--converted_answer_path ${CONVERTED_ANSWER_PATH} \
		--reference_model ${REFERENCE_MODEL} \
		--output_model ${CANDIDATE_MODEL} \
		--test_ids ../../data/test_ids/ \
		--save_path ${SAVE_PATH} \
		--pass_rate_result_path ${PASS_TARE_PATH} \
		--max_eval_threads 20 \
		--use_pass_rate true \
		--evaluate_times 4