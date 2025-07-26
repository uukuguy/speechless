
prepare_data:
	python prepare_verl_dataset.py \
	--config_name natural_reasoning_finance \
	--data_source /opt/local/datasets/llm-reasoning/natural_reasoning_finance \
	--dataset_name default \
	--output_dir ./data/natural_reasoning_finance

train_4b:
	bash run_rft_grpo_verl_4b.sh

train_8b:
	bash run_rft_grpo_verl_8b.sh

train_32b:
	bash run_rft_grpo_verl_32b.sh

merge:
	bash merge_verl_lora_fsdp.sh

vllm_serve:
	bash vllm_serve.sh

inference: 
	bash run_inference.sh
