help:
	@echo "Usage: make [prepare_data | dist_finetune]" 

prepare_data:
	python ./scripts/prepare_data.py

finetune:
	bash ./scripts/finetune_speechless_codellam_13b.sh

merge_peft_adapters:
	PYTHONPATH=${SPEECHLESS_ROOT} \
	python scripts/merge_peft_adapters.py \
		--base_model_name_or_path ${BASE_MODEL_PATH} \
		--peft_model_path ${CHECKPOINT_DIR} \
		--merged_model_name_or_path ${MERGED_MODEL_NAME} \

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

CUR_DIR=$(shell basename $(shell pwd))
TARGET_DIR=./sandbox/speechless.ai/${CUR_DIR}/

rsync_autodl_nm800_a40_2:
		rsync -rav --rsh='ssh -p 54192' * root@connect.neimeng.seetacloud.com:${TARGET_DIR}

rsync_autodl_nm799_a40_1:
		rsync -rav --rsh='ssh -p 45724' * root@connect.neimeng.seetacloud.com:${TARGET_DIR}

rsync_autodl_xbb092_4090_2:
	rsync -rav --rsh='ssh -p 14564' * root@connect.westb.seetacloud.com:${TARGET_DIR}

rsync_gpushare_a100_2:
	rsync -rav --rsh='ssh -p 59028' * root@i-2.gpushare.com:${TARGET_DIR}