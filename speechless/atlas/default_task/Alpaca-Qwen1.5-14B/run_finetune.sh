#!/bin/bash
bash run_singlenode.sh "python qwen1_5/run_qwen1_5.py \
--config finetune_qwen1_5_14b.yaml  \
--batch_size 1 \
--load_checkpoint ${PWD}/base_model \
--use_parallel True \
--run_mode finetune \
--auto_trans_ckpt True \
--train_data ${PWD}/datasets/train_dataset" \
rank_table_8.json [0,8] 8
