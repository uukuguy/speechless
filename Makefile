include Makefile.env

help:
	@echo "Usage: make [prepare_data | dist_finetune]" 

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

lm_eval_arc:
	python eval/run_lm_eval.py \
		--model hf-causal \
		--model_args pretrained=${MERGED_MODEL_NAME} \
		--tasks arc_challenge \
		--batch_size 1 \
		--limit 10 \
		--no_cache \
		--write_out \
		--output_path results/${TASK_NAME}/arc_challenge_25shot.json \
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
		--output_path results/${TASK_NAME}/hellaswag_10shot.json \
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
		--output_path results/${TASK_NAME}/mmlu_5shot.json \
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
		--output_path results/${TASK_NAME}/truthfullqa_0shot.json \
		--device cuda \
		--num_fewshot 0

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

BASENAME=$(shell basename $(shell pwd))
TARGET_DIR=./sandbox/speechless.ai/${BASENAME}/

# rsync -rav --exclude=.git --exclude=__pycache__ --rsh='ssh -p 45724' * root@connect.neimeng.seetacloud.com:${TARGET_DIR}
define push_to_remote
	@echo "push to remote: $(1), port: $(2), target: $(3)"
	rsync -rav \
		--exclude=.git \
		--exclude=__pycache__ \
		--rsh='ssh -p $(2)' \
		* \
		$(1):$(3)
endef

define pull_from_remote
	@echo "pull from remote: $(1), port: $(2), source: $(3)"
	rsync -rav \
		--exclude=.git \
		--exclude=__pycache__ \
		--rsh='ssh -p $(2)' \
		$(1):$(3) \
		.
endef

push_autodl_nm800_a40_2:
	# rsync -rav --exclude=.git --exclude=__pycache__ --rsh='ssh -p 54192' * root@connect.neimeng.seetacloud.com:${TARGET_DIR}
	$(call push_to_remote,root@connect.neimeng.seetacloud.com,54192,$(TARGET_DIR))

push_autodl_nm799_a40_1:
	# rsync -rav --exclude=.git --exclude=__pycache__ --rsh='ssh -p 45724' * root@connect.neimeng.seetacloud.com:${TARGET_DIR}
	$(call push_to_remote,root@connect.neimeng.seetacloud.com,45724,$(TARGET_DIR))

push_gpushare_a100_2:
	# rsync -rav --exclude=.git --exclude=__pycache__ --rsh='ssh -p 59028' * root@i-2.gpushare.com:${TARGET_DIR}
	$(call push_to_remote,root@i-2.gpushare.com,59028,$(TARGET_DIR))

push_gpushare_xn_4090x1:
	$(call push_to_remote,root@i-2.gpushare.com,48296,$(TARGET_DIR))

# ----- pull -----
pull_gpushare_hd_4090x1:
	$(call pull_from_remote,root@i-1.gpushare.com,55296,$(TARGET_DIR))