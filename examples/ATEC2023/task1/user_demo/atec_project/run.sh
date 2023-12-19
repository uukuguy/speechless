#!/bin/bash
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

TRAIN_FILE_ORIGIN=/home/admin/workspace/job/input/train_corpus.jsonl
DEV_FILE_ORIGIN=/home/admin/workspace/job/input/dev.jsonl

TRAIN_FILE=data/qa_data_samples_train.jsonl
DEV_FILE=data/qa_data_samples_dev.jsonl

MODEL_PATH=/adabench_mnt/llm/chatglm2-6b
OUTPUT=/home/admin/workspace/job/output/data

    # --max_steps 5000 \
    # --save_steps 5000 \

deepspeed --num_gpus=2 --master_port $MASTER_PORT src/main.py \
    --deepspeed src/deepspeed.json \
    --do_train \
    --train_file $TRAIN_FILE \
    --prompt_column query \
    --response_column answer \
    --overwrite_cache \
    --model_name_or_path $MODEL_PATH \
    --output_dir $OUTPUT \
    --overwrite_output_dir \
    --max_source_length 64 \
    --max_target_length 256 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --predict_with_generate \
    --logging_steps 10 \
    --save_strategy epoch \
    --num_train_epochs 1 \
    --learning_rate $LR \
    --pre_seq_len 128 \
    --fp16
