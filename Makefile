# include Makefile.env

MODELS_ROOT_DIR=/opt/local/llm_models/huggingface.co
# BASE_MODEL_PATH=${MODELS_ROOT_DIR}/Phind/Phind-CodeLlama-34B-v2
#BASE_MODEL_PATH=${MODELS_ROOT_DIR}/mistralai/Mistral-7B-v0.1
BASE_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-mistral-six-in-one-7b

# pass@1: 75.61
# TEST_MODEL_PATH=/opt/local/llm_models/huggingface.co/speechlessai/speechless-codellama-34b-v2.0
# pass@1: 70.73
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-codellama-34b-v1.9

# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-mistral-7b-v0.1
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-nl2sql-mistral-7b-v0.1
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/mistralai/Mistral-7B-v0.1
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/Phind/Phind-CodeLlama-34B-v2

# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-code-mistral-orca-7b-v1.0
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-code-mistral-7b-v1.0
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-tora-code-7b-v1.0
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-mistral-dolphin-orca-platypus-samantha-7b
TEST_MODEL_PATH=${MODELS_ROOT_DIR}/uukuguy/speechless-mistral-six-in-one-7b
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-tora-code-7b-v1.0

# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/Open-Orca/Mistral-7B-OpenOrca
#TEST_MODEL_PATH=${MODELS_ROOT_DIR}/Mistral-7B-OpenOrca-lora-merged
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/Mistral-7B-OpenOrca-r256-lora-merged
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/Mistral-7B-OpenOrca-r128-lora-merged

# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-code-mistral-7b-v1.1
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-code-mistral-7b-v1.0
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-code-mistral-orca-7b-v1.0

# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-tora-code-7b-v1.0
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/mistralai/Mistral-7B-v0.1

# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-codellama-34b-v2.1
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-codellama-34b-v2.0
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-tora-code-7b-v1.0

# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-reasoning-7b-v0
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-reasoning-7b-v0.1-tora
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-reasoning-7b-v0.2-mistral

#TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-reasoning-7b-v0.2-tora
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-reasoning-7b-v0.2-tora-cosine
#TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-reasoning-7b-v0.3-tora
#TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-reasoning-7b-v0.4-tora
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-agents-7b-v0.1-tora
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-agents-7b-v0.1-32k-mistral
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-reasoning-7b-v0.3-mistral
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/EleutherAI/llemma_7b
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-agents-7b-v0.1-tora
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-agents-7b-v0.1-32k-mistral
#TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-agents-7b-v0.1-32k-tora
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-agents-7b-v0.2-32k-tora
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-agents-7b-v0.2-32k-tora-794-steps
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-agents-7b-v0.2-32k-mistral
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-agents-7b-v0.2.2-32k-tora

#TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-tora-code-7b-v1.0
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-code-mistral-7b-v1.0

# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/mistralai/Mistral-7B-v0.1
#TEST_MODEL_PATH=${MODELS_ROOT_DIR}/llm_agents/tora-code-7b-v1.0
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-mistral-orca-7b-16k
#TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechless-agents-7b-v0.3-32k-mistral
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-mistral-orca-7b-32k-644-steps
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/Phind/Phind-CodeLlama-34B-v2
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/stabilityai/stablelm-3b-4e1t
#TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-coding-7b-16k-tora-1357-steps
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-coding-7b-16k-tora-2714-steps
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-coding-7b-16k-mistral-2714-steps

# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-coding-7b-16k-v1.3-mistral-1357-steps
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-coding-7b-16k-v1.3-mistral-2715-steps
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-coding-7b-16k-v1.3-mistral-4071-steps

# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-agents-7b-v0.2-32k-mistral

# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-coding-7b-orca2-1357-steps
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-coding-7b-orca2-1e
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-mistral-7b-dare-0.85

TASK_NAME=$(shell basename ${TEST_MODEL_PATH})

OUTPUT_DIR=./outputs
CHECKPOINT_DIR=${OUTPUT_DIR}/${TASK_NAME}/checkpoint-3500/adapter_model


help:
	@echo "Usage: make [prepare_data | finetune | inference | eval]" 

# -------------------- Train --------------------

prepare_data:
	python ./scripts/prepare_data.py

finetune_13b:
	bash ./scripts/finetune_speechless_codellam_13b.sh

finetune_34b:
	bash ./scripts/finetune_speechless_codellam_34b.sh

full_finetune_34b:
	bash ./scripts/full_finetune_speechless_codellam_34b.sh

finetune_v2.1:
	# bash ./scripts/finetune_speechless_codellam_34b_v2.1.sh
	cd tasks/speechless_codellama_34b_v2.1 && \
		bash ./finetune_speechless_codellama_34b_v2.1.sh

finetune_mistral_7b:
	# bash ./scripts/finetune_speechless_mistral_7b_v0.1.sh
	cd tasks/speechless_mistral_7b_v0.1 && \
		bash ./finetune_speechless_mistral_7b_v0.1.sh

merge_peft_adapters:
	PYTHONPATH=. \
	python scripts/merge_peft_adapters.py \
		--base_model_name_or_path ${BASE_MODEL_PATH} \
		--peft_model_path ${CHECKPOINT_DIR} \
		--merged_model_name_or_path ${TEST_MODEL_PATH} \


# -------------------- Inference --------------------

inference:
	PYTHONPATH=${SPEECHLESS_ROOT} \
	python inference.py \
		--base_model ${TEST_MODEL_PATH} \
		--test_file_path ${TEST_FILE} \

inference_with_lora:
	PYTHONPATH=${SPEECHLESS_ROOT} \
	python inference.py \
		--base_model ${BASE_MODEL_PATH} \
		--lora_weights ${CHECKPOINT_DIR} \
		--test_file_path ${TEST_FILE} \

# -------------------- lm-evaluation-harness --------------------

#https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard

lm_eval:
	bash ./speechless/eval/run_lm_eval.sh ${TEST_MODEL_PATH}


# -------------------- HumanEval --------------------
HUMANEVAL_OUTPUT_DIR=eval_results/human_eval/${TASK_NAME}

humaneval:
	PYTHONLIB=${SPEECHLESS_ROOT} \
	python -m speechless.eval.humaneval \
		--do_gen \
		--do_eval \
		--model ${TEST_MODEL_PATH} \
        --output_dir ${HUMANEVAL_OUTPUT_DIR}

# -------------------- Big Code Evaluation Harness --------------------
# BIGCODE_TASKS="humaneval,mbpp,multiple-py,multiple-java,multiple-js,multiple-cpp,multiple-rs,multiple-go,multiple-sh,multiple-jl"

# Don't run generation but benchmark groundtruth (useful for debugging)
# BIGCODE_CHECK_REFERENCES="--check_references"

# BITS="--load_in_8bit" 
# LIMIT="--limit 10"
# MAX_LENGTH_GENERATION=2048
# TEMPERATURE=0.2
# BITS="--load_in_8bit"
# PRECISION=bf16
# N_SAMPLES=1
# BATCH_SIZE=16

# bigcode_eval_gen:
# 	accelerate launch \
# 		--num_processes=2 \
# 		--num_machines=1 \
# 		--mixed_precision=${PRECISION} \
# 		--dynamo_backend=no \
# 		eval/bigcode_eval.py \
# 		--model ${TEST_MODEL_PATH} \
# 		${BITS} \
# 		--tasks ${BIGCODE_TASKS} \
# 		--limit 20 \
# 		--max_length_generation ${MAX_LENGTH_GENERATION} \
# 		--temperature ${TEMPERATURE} \
# 		--do_sample \
# 		--n_samples ${N_SAMPLES} \
# 		--batch_size ${BATCH_SIZE} \
# 		--precision ${PRECISION}\
# 		--trust_remote_code \
# 		--eval_results_dir eval_results/bigcode_eval/${TASK_NAME} \
# 		--generation_only \
# 		--save_generations \
# 		${BIGCODE_CHECK_REFERENCES}
bigcode_eval_gen:
	./eval/run_bigcode_eval_gen.sh ${TEST_MODEL_PATH}

bigcode_offical_gen:
	./eval/bigcode_offical_gen.sh ${TEST_MODEL_PATH}

# bigcode_eval_exec:
# 	accelerate launch  eval/bigcode_eval.py \
# 		--model ${TEST_MODEL_PATH}
# 		--tasks ${BIGCODE_TASKS} \
# 		--allow_code_execution  \
# 		--load_generations_path ${BIGCODE_SAVE_GENERATIONS_PATH} \
# 		--metric_output_path ${BIGCODE_METRIC_RESULTS_FILE} \

bigcode_eval:
	./eval/run_bigcode_eval.sh ${TEST_MODEL_PATH} multiple-py


# -------------------- MultiPL-E --------------------

# https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard

MULTIPL_E_RESULTS_DIR=eval_results/multipl_e/${TASK_NAME}

# MULTIPLE_E_LANG=mkdir -p ${MULTIPL_E_RESULTS_DIR} && \
# 	python eval/MultiPL-E/automodel.py \
# 		--name ${TEST_MODEL_PATH} \
# 		--root-dataset humaneval \
# 		--temperature 0.2 \
# 		--batch-size 20 \
# 		--completion-limit 20 \
# 		--output-dir-prefix ${MULTIPL_E_RESULTS_DIR} 
MULTIPLE_E_LANG=python eval/multiple.py \
		generate \
		--name ${TEST_MODEL_PATH} \
		--root-dataset humaneval \
		--temperature 0.2 \
		--batch-size 20 \
		--completion-limit 10 \
		--output-dir-prefix ${MULTIPL_E_RESULTS_DIR} 

# 34b ~ 56m
multipl_e_python:
	${MULTIPLE_E_LANG} \
		--lang py \

# 34b: ~ 1h 27m
multipl_e_java:
	${MULTIPLE_E_LANG} \
		--lang java \

# 34b: ~ 47m
multipl_e_js:
	${MULTIPLE_E_LANG} \
		--lang js \

# 34b: ~ 1h 40m
multipl_e_cpp:
	${MULTIPLE_E_LANG} \
		--lang cpp \

multipl_e_rust:
	${MULTIPLE_E_LANG} \
		--lang rs \

multipl_e_go:
	${MULTIPLE_E_LANG} \
		--lang go \

# multipl_e_gen: multipl_e_python multipl_e_java multipl_e_js multipl_e_cpp multipl_e_rust multipl_e_go
# 	@echo "multipl_e_gen done"

# multipl_e_gen:
# 	${MULTIPLE_E_LANG} \
# 		--langs py java js cpp rs go sh jl \

multiple_gen:
	# python eval/multiple_gen.py \
	# 	--do_generate \
	# 	-m ${TASK_NAME}  \
	# 	--langs py java js cpp rs go sh jl \
	# 	--completion_limit 5 && \
	# python eval/multiple_gen.py \
	# 	--do_convert \
	# 	--output_dir output_multiple_gen/${TASK_NAME}

	bash ./speechless/eval/run_multiple_gen.sh ${TEST_MODEL_PATH}

multiple_gen_50:
	bash speechless/eval/run_multiple_gen.sh ${TEST_MODEL_PATH} 50 20

multiple_50:
	bash speechless/eval/run_multiple.sh ${TEST_MODEL_PATH} 50

multiple:
	# python eval/multiple.py \
	# 	eval \
	# 	--results_dir output_multiple_gen/${TASK_NAME}/multiple && \
	# python eval/multiple.py \
	# 	results \
	# 	--results_dir output_multiple_gen/${TASK_NAME}/multiple

	bash ./speechless/eval/run_multiple.sh ${TEST_MODEL_PATH}


# multipl_e_eval:
# 	cd ${MULTIPL_E_RESULTS_DIR} && \
# 	bash ${PWD}/eval/run_multipl_e_eval.sh

# multipl_e_eval:
# 	python eval/multiple.py \
# 		eval \
# 		--results_dir ${MULTIPL_E_RESULTS_DIR}

# multipl_e_results:
# 	python ${PWD}/eval/MultiPL-E/pass_k.py -k 1 ${MULTIPL_E_RESULTS_DIR}/*

# multipl_e_results:
# 	python eval/multiple.py \
# 		results \
# 		--results_dir ${MULTIPL_E_RESULTS_DIR}



# -------------------- speechless.api.server --------------------
api_server:
	PYTHONPATH=${SPEECHLESS_ROOT} \
	python -m speechless.api.server \
		--model_name_or_path ${TEST_MODEL_PATH} \
		--model_family vllm \

include ../Makefile.remote