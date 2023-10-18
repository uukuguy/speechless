# include Makefile.env

MODELS_ROOT_DIR=/opt/local/llm_models/huggingface.co
# BASE_MODEL_PATH=${MODELS_ROOT_DIR}/Phind/Phind-CodeLlama-34B-v2
BASE_MODEL_PATH=${MODELS_ROOT_DIR}/mistralai/Mistral-7B-v0.1

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
#TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-mistral-six-in-one-7b
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

TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-codellama-34b-v2.1

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

# -------------------- HumanEval --------------------
HUMANEVAL_GEN_OUTPUT_FILE=eval_results/human_eval/${TASK_NAME}/humaneval_samples.jsonl

humaneval_gen:
	bash ./eval/run_humaneval_gen.sh \
		${TEST_MODEL_PATH} \
		${HUMANEVAL_GEN_OUTPUT_FILE}

	# PYTHONPATH=${PWD}/eval \

extract_code:
	python eval/extract_code.py ${HUMANEVAL_GEN_OUTPUT_FILE}

humaneval:
	python eval/run_humaneval.py \
		${HUMANEVAL_GEN_OUTPUT_FILE} \
		--problem_file ${PWD}/eval/datasets/openai_humaneval/HumanEval.jsonl.gz
		
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
		--completion-limit 20 \
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

multipl_e_gen:
	${MULTIPLE_E_LANG} \
		--langs py java js cpp rust go \

# multipl_e_eval:
# 	cd ${MULTIPL_E_RESULTS_DIR} && \
# 	bash ${PWD}/eval/run_multipl_e_eval.sh

multipl_e_eval:
	python eval/multiple.py \
		eval \
		--results_dir ${MULTIPL_E_RESULTS_DIR}

# multipl_e_results:
# 	python ${PWD}/eval/MultiPL-E/pass_k.py -k 1 ${MULTIPL_E_RESULTS_DIR}/*

multipl_e_results:
	python eval/multiple.py \
		results \
		--results_dir ${MULTIPL_E_RESULTS_DIR}


# -------------------- lm-evaluation-harness --------------------

#https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard

# lm_eval_arc:
# 	python eval/run_lm_eval.py \
# 		--model hf-causal \
# 		--model_args pretrained=${TEST_MODEL_PATH} \
# 		--tasks arc_challenge \
# 		--batch_size 16 \
# 		--limit 100 \
# 		--no_cache \
# 		--write_out \
# 		--output_path eval_results/lm_eval/${TASK_NAME}/arc_challenge_25shot.json \
# 		--device cuda \
# 		--num_fewshot 25

# lm_eval_hellaswag:
# 	python eval/run_lm_eval.py \
# 		--model hf-causal \
# 		--model_args pretrained=${TEST_MODEL_PATH} \
# 		--tasks hellaswag \
# 		--batch_size 16 \
# 		--limit 100 \
# 		--no_cache \
# 		--write_out \
# 		--output_path eval_results/lm_eval/${TASK_NAME}/hellaswag_10shot.json \
# 		--device cuda \
# 		--num_fewshot 10

# lm_eval_mmlu:
# 	python eval/run_lm_eval.py \
# 		--model hf-causal \
# 		--model_args pretrained=${TEST_MODEL_PATH} \
# 		--tasks hendrycksTest-* \
# 		--batch_size 16 \
# 		--limit 100 \
# 		--no_cache \
# 		--write_out \
# 		--output_path eval_results/lm_eval/${TASK_NAME}/mmlu_5shot.json \
# 		--device cuda \
# 		--num_fewshot 5

# lm_eval_truthfulqa:
# 	python eval/run_lm_eval.py \
# 		--model hf-causal \
# 		--model_args pretrained=${TEST_MODEL_PATH} \
# 		--tasks truthfulqa_mc \
# 		--batch_size 16 \
# 		--limit 100 \
# 		--no_cache \
# 		--write_out \
# 		--output_path eval_results/lm_eval/${TASK_NAME}/truthfullqa_0shot.json \
# 		--device cuda \
# 		--num_fewshot 0

# lmeval_open_llm: lm_eval_arc lm_eval_hellaswag lm_eval_mmlu lm_eval_truthfulqa
# 	@echo "lm_eval done"

lmeval:
	python eval/lmeval.py \
		--model hf-causal \
		--model_args pretrained=${TEST_MODEL_PATH} \
		--task \
			"arc_challenge|25|100" \
			"hellaswag|10|100" \
			"hendrycksTest-*|5|100" \
			"truthfulqa_mc|0|100" \
			"gsm8k|8|100" \
		--batch_size 4 \
		--no_cache \
		--write_out \
		--output_path eval_results/lm_eval/${TASK_NAME} 

# -------------------- vllm --------------------

# Run this command on the remote server which has the model.
# vllm:
# 	python -m vllm.entrypoints.openai.api_server \
# 		--host 0.0.0.0 \
# 		--port 8009 \
# 		--served-model-name ${TASK_NAME} \
# 		--dtype half \
# 		--trust-remote-code \
# 		--model ${TEST_MODEL_PATH}

api_server:
	PYTHONPATH=${PWD}/.. \
	python api/server.py \
		--model_name_or_path ${TEST_MODEL_PATH} \
		--model_family vllm \

include ../Makefile.remote

# # -------------------- Sync Remote --------------------

# BASENAME=$(shell basename $(shell pwd))
# TARGET_DIR=./sandbox/LLM/speechless.ai/${BASENAME}/

# # rsync -rav --exclude=.git --exclude=__pycache__ --rsh='ssh -p 45724' * root@connect.neimeng.seetacloud.com:${TARGET_DIR}
# define push_to_remote
# 	@echo "push to remote: $(1), port: $(2), target: $(3)"
# 	rsync -rav \
# 		--exclude=tmp \
# 		--exclude=outputs \
# 		--exclude=saved_models \
# 		--exclude=.git \
# 		--exclude=__pycache__ \
# 		--exclude=tasks \
# 		--exclude=eval_results \
# 		--rsh='ssh -p $(2)' \
# 		* \
# 		$(1):$(3)
# endef

# define pull_from_remote
# 	@echo "pull from remote: $(1), port: $(2), source: $(3), subdir: $(4)"
# 	rsync -rav \
# 		--exclude=tmp \
# 		--exclude=outputs \
# 		--exclude=saved_models \
# 		--exclude=.git \
# 		--exclude=__pycache__ \
# 		--exclude=checkpoint* \
# 		--rsh='ssh -p $(2)' \
# 		$(1):$(3)/$(4)/ \
# 		$(4)/
# endef

# # ----- push to remote -----
# push_autodl_nm800_a40x2:
# 	$(call push_to_remote,root@connect.neimeng.seetacloud.com,54192,$(TARGET_DIR))

# push_autodl_nm799_a40x2:
# 	$(call push_to_remote,root@connect.neimeng.seetacloud.com,45724,$(TARGET_DIR))

# push_gpushare_h800x2:
# 	$(call push_to_remote,root@i-1.gpushare.com,5893,$(TARGET_DIR))

# push_gpushare_a800x2:
# 	$(call push_to_remote,root@i-1.gpushare.com,12391,$(TARGET_DIR))

# push_gpushare_a100x4:
# 	$(call push_to_remote,root@i-1.gpushare.com,2917,$(TARGET_DIR))


# # ----- pull from remote -----
# pull_autodl_nm799_a40x2_eval_results:
# 	$(call pull_from_remote,root@connect.neimeng.seetacloud.com,45724,$(TARGET_DIR),eval_results)

# pull_gpushare_h800x2_eval_results:
# 	$(call pull_from_remote,root@i-1.gpushare.com,5893,$(TARGET_DIR),eval_results)

# pull_gpushare_a800x2_eval_results:
# 	$(call pull_from_remote,root@i-1.gpushare.com,12391,$(TARGET_DIR),eval_results)

# pull_gpushare_a100x4_eval_results:
# 	$(call pull_from_remote,root@i-1.gpushare.com,2917,$(TARGET_DIR),eval_results)
