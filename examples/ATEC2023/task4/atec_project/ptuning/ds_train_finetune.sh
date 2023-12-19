
LR=1e-4

MASTER_PORT=$(shuf -n 1 -i 10000-65535)

deepspeed --num_gpus=3 --master_port $MASTER_PORT ptuning/main.py \
    --deepspeed ptuning/deepspeed.json \
    --do_train \
    --train_file utils/build_datasets/training.json \
    --validation_file utils/build_datasets/training.json \
    --prompt_column instruction \
    --response_column output \
    --overwrite_cache \
    --cache_dir cache \
    --model_name_or_path /adabench_mnt/llm/chatglm2-6b \
    --output_dir /home/admin/workspace/job/output/data/full-parameter-model/ \
    --overwrite_output_dir \
    --max_source_length 1024 \
    --max_target_length 32 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --predict_with_generate \
    --max_steps 200 \
    --logging_steps 10 \
    --save_steps 50 \
    --learning_rate $LR \
    --fp16

