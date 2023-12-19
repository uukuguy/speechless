import os

def make_os_line(base_model_dir, outputdir):
    str = 'CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nnodes=1 --nproc-per-node=1 main.py --do_train ' \
        '--train_file ./data/train.json ' \
        '--validation_file ./data/val.json ' \
        '--preprocessing_num_workers 10 ' \
        '--prompt_column input ' \
        '--response_column output ' \
        '--overwrite_cache ' \
        '--model_name_or_path %s ' \
        '--output_dir %s ' \
        '--overwrite_output_dir ' \
        '--max_source_length 512 ' \
        '--max_target_length 128 ' \
        '--per_device_train_batch_size 1 ' \
        '--per_device_eval_batch_size 1 '\
        '--gradient_accumulation_steps 16 ' \
        '--predict_with_generate ' \
        '--max_steps 11 '\
        '--logging_steps 10 ' \
        '--save_steps 10 ' \
        '--learning_rate 2e-2 ' \
        '--pre_seq_len 128'%(base_model_dir, outputdir)

    os.system(str)
