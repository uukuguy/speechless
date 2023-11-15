MODELS_ROOT_DIR=/opt/local/llm_models/huggingface.co
#TOOLLLAMA_MODEL_PATH=${MODELS_ROOT_DIR}/ToolBench/ToolLLaMA-2-7b-v2

TOOLLLAMA_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-tools-7b-32k-tora-3e-1435-steps
# TOOLLLAMA_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-tools-7b-32k-tora-3e-2-1435-steps
#TOOLLLAMA_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-tools-7b-32k-tora-3e-2-2870-steps
# TOOLLLAMA_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-tools-7b-32k-mistral-2871-steps
# TOOLLLAMA_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-tools-7b-32k-v0.5-2871-steps
# TOOLLLAMA_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-tools-7b-32k-v0.5-5742-steps
# TOOLLLAMA_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-tools-7b-v0.1-32k-mistral-5743-steps
# TOOLLLAMA_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-tools-7b-multi-rounds-32k-tora-train0.15-861-steps
RETRIEVAL_MODEL_PATH=${MODELS_ROOT_DIR}/ToolBench/ToolBench_IR_bert_based_uncased

TOOLBENCH_DATA_DIR=/opt/local/datasets/toolbench_data
TOOL_ROOT_DIR=${TOOLBENCH_DATA_DIR}/toolenv/tools/
CORPUS_TSV_PATH=${TOOLBENCH_DATA_DIR}/retrieval/G1/corpus.tsv
INPUT_QUERY_FILE=${TOOLBENCH_DATA_DIR}/test_instruction/G1_instruction.json
RETRIEVED_API_NUMS=5
METHOD=DFS_woFilter_w2

SAMPLING_ARGS=--temperature 0.5 --max_tokens 512 --top_p 1.0 --top_k 50
EXEC_ARGS=--max_observation_length 512 --observ_compress_method truncate --method ${METHOD}

RETRIEVAL_ARGS="--corpus_tsv_path ${CORPUS_TSV_PATH} --retrieval_model_path ${RETRIEVAL_MODEL_PATH} --retrieved_api_nums ${RETRIEVED_API_NUMS} "


MODEL_NAME=$(shell basename ${TOOLLLAMA_MODEL_PATH})
# OUTPUT_DIR=${PWD}/outputs/${MODEL_NAME}.${METHOD}
OUTPUT_DIR=${PWD}/outputs/${MODEL_NAME}


toolbench_G1_instruction:
	PYTHONPATH=${SPEECHLESS_ROOT} \
	python rapidapi/run_rapidapi.py \
		--tool_root_dir ${TOOL_ROOT_DIR} \
		--backbone_model vllm \
		--model_path ${TOOLLLAMA_MODEL_PATH} \
		${EXEC_ARGS} \
		${SAMPLING_ARGS} \
		--input_query_file ${TOOLBENCH_DATA_DIR}/test_instruction/G1_instruction.json \
		--output_answer_file ${OUTPUT_DIR}/G1_instruction \
		--toolbench_key ${TOOLBENCH_KEY}


toolbench_G1_tool:
	PYTHONPATH=${SPEECHLESS_ROOT} \
	python rapidapi/run_rapidapi.py \
		--tool_root_dir ${TOOL_ROOT_DIR} \
		--backbone_model vllm \
		--model_path ${TOOLLLAMA_MODEL_PATH} \
		${EXEC_ARGS} \
		${SAMPLING_ARGS} \
		--input_query_file ${TOOLBENCH_DATA_DIR}/test_instruction/G1_tool.json \
		--output_answer_file ${OUTPUT_DIR}/G1_tool \
		--toolbench_key ${TOOLBENCH_KEY}


toolbench_G1_category:
	PYTHONPATH=${SPEECHLESS_ROOT} \
	python rapidapi/run_rapidapi.py \
		--tool_root_dir ${TOOL_ROOT_DIR} \
		--backbone_model vllm \
		--model_path ${TOOLLLAMA_MODEL_PATH} \
		${EXEC_ARGS} \
		${SAMPLING_ARGS} \
		--input_query_file ${TOOLBENCH_DATA_DIR}/test_instruction/G1_category.json \
		--output_answer_file ${OUTPUT_DIR}/G1_category \
		--toolbench_key ${TOOLBENCH_KEY}


toolbench_G2_instruction:
	PYTHONPATH=${SPEECHLESS_ROOT} \
	python rapidapi/run_rapidapi.py \
		--tool_root_dir ${TOOL_ROOT_DIR} \
		--backbone_model vllm \
		--model_path ${TOOLLLAMA_MODEL_PATH} \
		${EXEC_ARGS} \
		${SAMPLING_ARGS} \
		--input_query_file ${TOOLBENCH_DATA_DIR}/test_instruction/G2_instruction.json \
		--output_answer_file ${OUTPUT_DIR}/G2_instruction \
		--toolbench_key ${TOOLBENCH_KEY}


toolbench_G2_category:
	PYTHONPATH=${SPEECHLESS_ROOT} \
	python rapidapi/run_rapidapi.py \
		--tool_root_dir ${TOOL_ROOT_DIR} \
		--backbone_model vllm \
		--model_path ${TOOLLLAMA_MODEL_PATH} \
		${EXEC_ARGS} \
		${SAMPLING_ARGS} \
		--input_query_file ${TOOLBENCH_DATA_DIR}/test_instruction/G2_category.json \
		--output_answer_file ${OUTPUT_DIR}/G2_category \
		--toolbench_key ${TOOLBENCH_KEY}


toolbench_G3_instruction:
	PYTHONPATH=${SPEECHLESS_ROOT} \
	python rapidapi/run_rapidapi.py \
		--tool_root_dir ${TOOL_ROOT_DIR} \
		--backbone_model vllm \
		--model_path ${TOOLLLAMA_MODEL_PATH} \
		${EXEC_ARGS} \
		${SAMPLING_ARGS} \
		--input_query_file ${TOOLBENCH_DATA_DIR}/test_instruction/G3_instruction.json \
		--output_answer_file ${OUTPUT_DIR}/G3_instruction \
		--toolbench_key ${TOOLBENCH_KEY}


toolbench_speechless:
	PYTHONPATH=${SPEECHLESS_ROOT} \
	python rapidapi/run_rapidapi.py \
		--tool_root_dir ${TOOL_ROOT_DIR} \
		--backbone_model toolllama \
		--model_path ${TOOLLLAMA_MODEL_PATH} \
		${EXEC_ARGS} \
		${SAMPLING_ARGS} \
		--input_query_file ${INPUT_QUERY_FILE} \
		--output_answer_file ${OUTPUT_DIR} \
		--toolbench_key ${TOOLBENCH_KEY}


toolbench_openai:
	PYTHONPATH=${SPEECHLESS_ROOT} \
	python rapidapi/run_rapidapi.py \
		--tool_root_dir ${TOOL_ROOT_DIR} \
		--backbone_model chatgpt_function \
		--openai_key ${OPENAI_KEY} \
		${EXEC_ARGS} \
		${SAMPLING_ARGS} \
		--input_query_file ${INPUT_QUERY_FILE} \
		--output_answer_file chatgpt_dfs_inference_result \
		--toolbench_key ${TOOLBENCH_KEY}


toolbench_opendomain:
	PYTHONPATH=${SPEECHLESS_ROOT} \
	python rapidapi/run_rapidapi.py \
		--tool_root_dir ${TOOL_ROOT_DIR} \
		--backbone_model toolllama \
		--model_path ${TOOLLLAMA_MODEL_PATH} \
		${RETRIEVAL_ARGS} \
		${EXEC_ARGS} \
		${SAMPLING_ARGS} \
		--input_query_file ${INPUT_QUERY_FILE} \
		--output_answer_file ${PWD}/toolllama_dfs_open_domain_inference_result \
		--toolbench_key ${TOOLBENCH_KEY}


toolbench_eval_prepare:
	cd tooleval
	export RAW_ANSWER_PATH=${TOOLBENCH_DATA_DIR}/reproduction_data/model_predictions/
	export CONVERTED_ANSWER_PATH=${TOOLBENCH_DATA_DIR}/reproduction_data/model_predictions_converted/
	export MODEL_NAME=speechless-tools-7b-32k-v0.15_dfs
	export METHOD=DFS_woFilter_w2
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


toolbench_pass_rate:
	cd tooleval && \
	export CONVERTED_ANSWER_PATH=${TOOLBENCH_DATA_DIR}/reproduction_data/model_predictions_converted/ \
	export SAVE_PATH=pass_rate_results \
	export CANDIDATE_MODEL=speechless-tools-7b-32k-v0.15_dfs \
	python eval_pass_rate.py \
		--converted_answer_path ${CONVERTED_ANSWER_PATH} \
		--save_path ${SAVE_PATH} \
		--reference_model ${CANDIDATE_MODEL} \
		--test_ids ${TOOLBENCH_DATA_DIR}/data/test_query_ids/ \
		--max_eval_threads 20 \
		--evaluate_times 4

toolbench_win_rate:
	cd tooleval
	export CONVERTED_ANSWER_PATH=${TOOLBENCH_DATA_DIR}/reproduction_data/model_predictions_converted/
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