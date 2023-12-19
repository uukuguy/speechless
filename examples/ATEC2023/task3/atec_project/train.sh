PRE_SEQ_LEN=128
LR=2e-2
NUM_GPUS=1

#    --max_steps 750 \
#    --save_steps 50 \

CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nnodes=1 --nproc-per-node=$NUM_GPUS main.py \
    --do_train \
    --train_file data/train.json \
    --validation_file data/dev.json \
    --preprocessing_num_workers 10 \
    --prompt_column input \
    --response_column output \
    --overwrite_cache \
    --model_name_or_path /adabench_mnt/llm/chatglm2-6b \
    --output_dir /home/admin/workspace/job/output/data/ptuing-chatglm2-6b-pt-$PRE_SEQ_LEN-$LR \
    --overwrite_output_dir \
    --max_source_length 512 \
    --max_target_length 64 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --logging_steps 10 \
    --eval_steps 50 \
    --num_train_epochs 3 \
    --save_strategy epoch \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    # --quantization_bit 4
