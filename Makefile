include Makefile.env

help:
	@echo "Usage: make [prepare_data | finetune | inference | eval]" 

# -------------------- Train --------------------

prepare_data:
	python ./scripts/prepare_data.py

finetune:
	bash ./scripts/finetune_speechless_codellam_13b.sh

merge_peft_adapters:
	PYTHONPATH=. \
	python scripts/merge_peft_adapters.py \
		--base_model_name_or_path ${BASE_MODEL_PATH} \
		--peft_model_path ${CHECKPOINT_DIR} \
		--merged_model_name_or_path ${MERGED_MODEL_NAME} \


# -------------------- Inference --------------------

inference:
	PYTHONPATH=${SPEECHLESS_ROOT} \
	python inference.py \
		--base_model ${MERGED_MODEL_NAME} \
		--test_file_path ${TEST_FILE} \

inference_with_lora:
	PYTHONPATH=${SPEECHLESS_ROOT} \
	python inference.py \
		--base_model ${BASE_MODEL_PATH} \
		--lora_weights ${CHECKPOINT_DIR} \
		--test_file_path ${TEST_FILE} \


# -------------------- HumanEval --------------------

human_eval_gen:
	bash ./eval/run_human_eval_gen.sh

human_eval_speechless_13b:
	PYTHONPATH=${PWD}/eval \
	python eval/run_human_eval.py \
		./eval_results/human_eval/speechless-codellama-airoboros-orca-platypus-13b.local_vllm/human_eval_samples.jsonl \
		--problem_file ${PWD}/eval/datasets/openai_humaneval/HumanEval.jsonl.gz

human_eval_speechless_34b:
	PYTHONPATH=${PWD}/eval \
	python eval/run_human_eval.py \
		./eval_results/human_eval/speechless-codellama-34b-v1.0_vllm/human_eval_samples.jsonl \
		--problem_file ${PWD}/eval/datasets/openai_humaneval/HumanEval.jsonl.gz

human_eval_phi:
	PYTHONPATH=${PWD}/eval \
	python eval/run_human_eval.py \
		./eval_results/human_eval/phi-1_5/human_eval_samples.jsonl \
		--problem_file ${PWD}/eval/datasets/openai_humaneval/HumanEval.jsonl.gz


# -------------------- MultiPL-E --------------------

# https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard

MULTIPL_E_RESULTS_DIR=eval_results/multipl_e
#MULTIPL_E_NAME=$(shell basename ${MERGED_MODEL_NAME})

MULTIPLE_E_LANG=mkdir -p ${MULTIPL_E_RESULTS_DIR} && \
	python eval/MultiPL-E/automodel.py \
		--name ${MERGED_MODEL_NAME} \
		--root-dataset humaneval \
		--temperature 0.2 \
		--batch-size 20 \
		--completion-limit 20 \
		--output-dir-prefix ${MULTIPL_E_RESULTS_DIR} 

multipl_e_python:
	${MULTIPLE_E_LANG} \
		--lang py \

multipl_e_java:
	${MULTIPLE_E_LANG} \
		--lang java \

multipl_e_js:
	${MULTIPLE_E_LANG} \
		--lang js \

multipl_e_cpp:
	${MULTIPLE_E_LANG} \
		--lang cpp \

multipl_e_rust:
	${MULTIPLE_E_LANG} \
		--lang rs \

multipl_e_gen: multipl_e_python multipl_e_java multipl_e_js multipl_e_cpp multipl_e_rust
	@echo "multipl_e_gen done"

multipl_e_eval:
	cd ${MULTIPL_E_RESULTS_DIR} && \
	bash ${PWD}/eval/run_multipl_e_eval.sh

multipl_e_results:
	python ${PWD}/eval/MultiPL-E/pass_k.py -k 1 ${MULTIPL_E_RESULTS_DIR}/*


# -------------------- lm-evaluation-harness --------------------

#https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard

lm_eval_arc:
	python eval/run_lm_eval.py \
		--model hf-causal \
		--model_args pretrained=${MERGED_MODEL_NAME} \
		--tasks arc_challenge \
		--batch_size 1 \
		--limit 10 \
		--no_cache \
		--write_out \
		--output_path eval_results/lm_eval/${TASK_NAME}/arc_challenge_25shot.json \
		--device cuda \
		--num_fewshot 25

lm_eval_hellaswag:
	python eval/run_lm_eval.py \
		--model hf-causal \
		--model_args pretrained=${MERGED_MODEL_NAME} \
		--tasks hellaswag \
		--batch_size 1 \
		--limit 10 \
		--no_cache \
		--write_out \
		--output_path eval_results/lm_eval/${TASK_NAME}/hellaswag_10shot.json \
		--device cuda \
		--num_fewshot 100

lm_eval_mmlu:
	python eval/run_lm_eval.py \
		--model hf-causal \
		--model_args pretrained=${MERGED_MODEL_NAME} \
		--tasks hendrycksTest-* \
		--batch_size 1 \
		--limit 10 \
		--no_cache \
		--write_out \
		--output_path eval_results/lm_eval/${TASK_NAME}/mmlu_5shot.json \
		--device cuda \
		--num_fewshot 5

lm_eval_truthfulqa:
	python eval/run_lm_eval.py \
		--model hf-causal \
		--model_args pretrained=${MERGED_MODEL_NAME} \
		--tasks truthfulqa_mc \
		--batch_size 1 \
		--limit 10 \
		--no_cache \
		--write_out \
		--output_path eval_results/lm_eval/${TASK_NAME}/truthfullqa_0shot.json \
		--device cuda \
		--num_fewshot 0


# -------------------- Sync Remote --------------------

BASENAME=$(shell basename $(shell pwd))
TARGET_DIR=./sandbox/LLM/speechless.ai/${BASENAME}/

# rsync -rav --exclude=.git --exclude=__pycache__ --rsh='ssh -p 45724' * root@connect.neimeng.seetacloud.com:${TARGET_DIR}
define push_to_remote
	@echo "push to remote: $(1), port: $(2), target: $(3)"
	rsync -rav \
		--exclude=tmp \
		--exclude=outputs \
		--exclude=saved_models \
		--exclude=.git \
		--exclude=__pycache__ \
		--rsh='ssh -p $(2)' \
		* \
		$(1):$(3)
endef

define pull_from_remote
	@echo "pull from remote: $(1), port: $(2), source: $(3)"
	rsync -rav \
		--exclude=tmp \
		--exclude=outputs \
		--exclude=saved_models \
		--exclude=.git \
		--exclude=__pycache__ \
		--rsh='ssh -p $(2)' \
		$(1):$(3) \
		.
endef

# ----- push to remote -----
push_autodl_nm800_a40_2:
	$(call push_to_remote,root@connect.neimeng.seetacloud.com,54192,$(TARGET_DIR))

push_autodl_nm799_a40_1:
	$(call push_to_remote,root@connect.neimeng.seetacloud.com,45724,$(TARGET_DIR))

push_gpushare_a100_2:
	$(call push_to_remote,root@i-2.gpushare.com,59028,$(TARGET_DIR))

push_gpushare_xn_4090x1:
	$(call push_to_remote,root@i-2.gpushare.com,48296,$(TARGET_DIR))

# ----- pull from remote -----
pull_autodl_nm799_a40_1:
	$(call pull_from_remote,root@connect.neimeng.seetacloud.com,45724,$(TARGET_DIR))

pull_gpushare_hd_4090x1:
	$(call pull_from_remote,root@i-1.gpushare.com,55296,$(TARGET_DIR))
