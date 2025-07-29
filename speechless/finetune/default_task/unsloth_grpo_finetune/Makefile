prepare_data:
	@echo "Preate Data"

train:
	PYTHONPATH=${SPEECHLESS_ROOT} python unsloth_grpo_finetune.py

merge:
	bash merge_unsloth_lora.sh

vllm_serve:
	bash vllm_serve.sh

inference: 
	bash run_inference.sh
