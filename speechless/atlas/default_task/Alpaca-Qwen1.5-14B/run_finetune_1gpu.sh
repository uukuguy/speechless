#!/bin/bash
python qwen1_5/run_qwen1_5.py \
	--config finetune_qwen1_5_14b.yaml  \
	--load_checkpoint ${PWD}/base_model \
	--run_mode finetune \
	--train_data ${PWD}/datasets/train_dataset
