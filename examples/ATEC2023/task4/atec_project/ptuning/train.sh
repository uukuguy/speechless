PRE_SEQ_LEN=64
LR=2e-2
NUM_GPUS=2

    #--max_steps 200 \
    #--save_steps 100 \

torchrun --standalone --nnodes=1 --nproc-per-node=$NUM_GPUS ptuning/main.py \
    --do_train \
    --train_file utils/build_datasets/training.json \
    --validation_file utils/build_datasets/training.json \
    --preprocessing_num_workers 10 \
    --prompt_column instruction \
    --response_column output \
    --overwrite_cache \
    --cache_dir cache \
    --model_name_or_path /adabench_mnt/llm/chatglm2-6b \
    --output_dir /home/admin/workspace/job/output/data/ptuning-model \
    --overwrite_output_dir \
    --max_source_length 1024 \
    --max_target_length 32 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --logging_steps 10 \
    --num_train_epochs 1 \
    --save_strategy epoch \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
#    --quantization_bit 4

