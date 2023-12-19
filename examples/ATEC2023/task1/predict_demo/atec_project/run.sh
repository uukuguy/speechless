#!/bin/sh
#****************************************************************#
# ScriptName: evaluate_finetune.sh
# Author: $SHTERM_REAL_USER@alibaba-inc.com
# Create Date: 2023-11-21 17:59
# Modify Author: $SHTERM_REAL_USER@alibaba-inc.com
# Modify Date: 2023-11-21 18:00
# Function: 
#***************************************************************#
DIR1="`dirname $BASH_SOURCE`"
MYDIR=`readlink -f "$DIR1"`
cd ${MYDIR}
pwd
source env/bin/activate
export PYTHONHOME=${MYDIR}/env
export PYTHONPATH=${MYDIR}/src:$PYTHONPATH
echo "start python code"

ls -lt /home/admin/workspace/
ls -lt /home/admin/workspace/job
ls -lt /home/admin/workspace/job/input

LR=1e-5

MASTER_PORT=$(shuf -n 1 -i 10000-65535)
export WANDB_DISABLED="true"

TEST_FILE_ORIGIN=/home/admin/workspace/job/input/test.jsonl

#TEST_FILE=data/test.jsonl

MODEL_PATH=/adabench_mnt/llm/chatglm2-6b
OUTPUT=/home/admin/workspace/job/output/data

CHECKPOINT_PATH=model/checkpoint-5000/
MODEL_PATH=/adabench_mnt/llm/chatglm2-6b
OUTPUT=/home/admin/workspace/job/output/predictions
NUM_GPUS=1

torchrun --standalone --nnodes=1 --nproc-per-node=$NUM_GPUS src/main.py \
    --do_predict \
    --test_file $TEST_FILE_ORIGIN  \
    --overwrite_cache \
    --prompt_column question \
    --model_name_or_path $MODEL_PATH \
    --ptuning_checkpoint $CHECKPOINT_PATH  \
    --output_dir $OUTPUT \
    --overwrite_output_dir \
    --max_source_length 64 \
    --max_target_length 256 \
    --per_device_eval_batch_size 8 \
    --predict_with_generate \
    --pre_seq_len 128 \
    --fp16_full_eval
