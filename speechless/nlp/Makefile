MODEL_ROOT_DIR="/opt/local/llm_models/huggingface.co"
MODEL_PATH="${MODEL_ROOT_DIR}/google-bert/bert-base-chinese"
TRAIN_FILE_PATH="./data/AFAC2024-Track1-api-classifer-train.jsonl"
VAL_FILE_PATH="./data/AFAC2024-Track1-api-classifer-val.jsonl"
OUTPUTS_DIR="./outputs"
NUM_TRAIN_EPOCHS=15

help:
	@echo "train_apis - Train the model on the API classification task"
	@echo "run_apis - Run the model on the API classification task"

train_apis:
	PYTHONPATH=${SPEECHLESS_ROOT} \
	python run_classification.py \
		--model_name_or_path ${MODEL_PATH} \
		--train_file ${TRAIN_FILE_PATH} \
		--validation_file ${VAL_FILE_PATH} \
		--shuffle_train_dataset \
		--metric_name f1 \
		--text_column_names text \
		--label_column_name label \
		--do_train \
		--do_eval \
		--logging_steps 1 \
		--eval_steps 10 \
		--eval_strategy steps \
		--save_strategy epoch \
		--save_total_limit ${NUM_TRAIN_EPOCHS} \
		--max_seq_length 512 \
		--per_device_train_batch_size 32 \
		--learning_rate 7e-7 \
		--lr_scheduler_type cosine \
		--warmup_ratio 0.03 \
		--num_train_epochs ${NUM_TRAIN_EPOCHS} \
		--report_to tensorboard \
		--overwrite_output_dir \
		--output_dir ${OUTPUTS_DIR}

TARGET_MODEL_PATH="outputs/checkpoint-2985"
TEST_A_FILE_PATH="./data/AFAC2024-Track1-api-classifer-test-queries.jsonl"
run_apis_test_queries:
	PYTHONPATH=${SPEECHLESS_ROOT} \
	python run_classification.py \
		--model_name_or_path ${TARGET_MODEL_PATH} \
		--train_file ${TRAIN_FILE_PATH} \
		--validation_file ${VAL_FILE_PATH} \
		--test_file ${TEST_A_FILE_PATH} \
		--shuffle_train_dataset \
		--metric_name f1 \
		--text_column_names text \
		--label_column_name label \
		--do_predict \
		--logging_steps 1 \
		--max_seq_length 512 \
		--per_device_train_batch_size 32 \
		--learning_rate 7e-5 \
		--num_train_epochs ${NUM_TRAIN_EPOCHS} \
		--report_to tensorboard \
		--overwrite_output_dir \
		--output_dir ${OUTPUTS_DIR}

TRAIN_QUERIES_FILE_PATH="./data/AFAC2024-Track1-api-classifer-train-queries.jsonl"
run_apis_train_queries:
	PYTHONPATH=${SPEECHLESS_ROOT} \
	python run_classification.py \
		--model_name_or_path ${TARGET_MODEL_PATH} \
		--train_file ${TRAIN_FILE_PATH} \
		--validation_file ${TRAIN_QUERIES_FILE_PATH} \
		--test_file ${TRAIN_QUERIES_FILE_PATH} \
		--shuffle_train_dataset \
		--metric_name f1 \
		--text_column_names text \
		--label_column_name label \
		--do_eval \
		--do_predict \
		--logging_steps 1 \
		--max_seq_length 512 \
		--per_device_train_batch_size 32 \
		--learning_rate 7e-5 \
		--num_train_epochs ${NUM_TRAIN_EPOCHS} \
		--report_to tensorboard \
		--overwrite_output_dir \
		--output_dir ${OUTPUTS_DIR}
