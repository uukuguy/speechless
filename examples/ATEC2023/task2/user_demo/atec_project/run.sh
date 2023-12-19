#!/bin/bash
set -x
DIR1="`dirname $BASH_SOURCE`"
MYDIR=`readlink -f "$DIR1"`
cd ${MYDIR}
pwd
source env/bin/activate
export PYTHONHOME=${MYDIR}/env
export PYTHONPATH=${MYDIR}/src:$PYTHONPATH
echo "start python code"

LR=6e-6
DATE=`date +%s`
MASTER_PORT=8888
TRAIN_FILE_RAW=/home/admin/workspace/job/input/train.jsonl
DEV_FILE_RAW=/home/admin/workspace/job/input/dev.jsonl

TRAIN_FILE=/home/admin/workspace/job/input/train_tl_conv.jsonl
DEV_FILE=/home/admin/workspace/job/input/dev_tl_conv.jsonl 

python src/convert_to_conv_data.py --orig_data $TRAIN_FILE_RAW --write_data $TRAIN_FILE --dataset_name tl_train
python src/convert_to_conv_data.py --orig_data $DEV_FILE_RAW --write_data $DEV_FILE --dataset_name tl_dev

ls -lt /home/admin/workspace/job/input

    #--save_steps 1000 \

#PRE_SEQ_LEN=128
MODEL_PATH=/adabench_mnt/llm/chatglm2-6b
OUTPUT=/home/admin/workspace/job/output/data
deepspeed --master_port $MASTER_PORT src/main.py \
    --quantization_bit 8 \
    --deepspeed src/deepspeed.json \
    --fp16 True \
    --do_train \
    --train_file $TRAIN_FILE \
    --validation_file $DEV_FILE \
    --prompt_column conversations \
    --overwrite_cache \
    --model_name_or_path $MODEL_PATH \
    --output_dir $OUTPUT \
    --overwrite_output_dir \
    --max_length 3000 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --predict_with_generate \
    --save_strategy epoch \
    --num_train_epochs 10 \
    --logging_steps 20 \
    --learning_rate $LR \
    --do_eval False \
    --save_total_limit 2
