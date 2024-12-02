#!/usr/bin/env python3
"""
## Training
Usage: python -m speechless.finetune.finetune_qlora \
    --do_train \
    --model_name_or_path <model_name_or_path> \
    --dataset <dataset> \
    --output_dir <output_dir> \
    --overwrite_output_dir \
    --per_device_train_batch_size <per_device_train_batch_size> \
    --per_device_eval_batch_size <per_device_eval_batch_size> \
    --learning_rate <learning_rate> \
    --lora_rank <lora_rank> \
    --lora_alpha <lora_alpha> \
    --num_train_epochs <num_train_epochs> \
    --training_strategy <training_strategy> \
    --training_steps <training_steps> \
    --warmup_steps <warmup_steps> \
    --eval_steps <eval_steps> \
    --logging_steps <logging_steps> \
    --save_steps <save_steps> \
    --seed <seed> \

## Predicting
Usage: python -m speechless.finetune.finetune_qlora \
    --do_predict \
    --model_name_or_path <model_name_or_path> \
    --dataset <dataset> \
    --output_dir <output_dir> \
    --per_device_eval_batch_size <per_device_eval_batch_size> \
    --seed <seed> \
    
"""
import os, json
from transformers import set_seed, Seq2SeqTrainer
import bitsandbytes as bnb
from loguru import logger

from speechless.utils.model_utils import get_accelerate_model, print_trainable_parameters
from speechless.utils.tokenizer_utils import get_tokenizer
from speechless.finetune.dataset_utils import make_data_module
from speechless.finetune.callbacks import LoggingCallback, CleanMemoryCallback, SavePeftModelCallback
from speechless.finetune.finetune_qlora_arguments import get_args

def get_last_checkpoint(checkpoint_dir):
    if os.path.isdir(checkpoint_dir):
        is_completed = os.path.exists(os.path.join(checkpoint_dir, 'completed'))
        if is_completed: return None, True # already finished
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if os.path.isdir(os.path.join(checkpoint_dir, filename)) and filename.startswith('checkpoint'):
                max_step = max(max_step, int(filename.replace('checkpoint-', '')))
        if max_step == 0: return None, is_completed # training started, but no checkpoint
        checkpoint_dir = os.path.join(checkpoint_dir, f'checkpoint-{max_step}')
        logger.info(f"Found a previous checkpoint at: {checkpoint_dir}")
        return checkpoint_dir, is_completed # checkpoint found!
    return None, False # first training

def finetune():
    args, model_args, data_args, training_args, remaining_args = get_args()
    logger.info(f"Arguments: {args}")
    logger.info(f"Model Arguments: {model_args}")
    logger.info(f"Data Arguments: {data_args}")
    logger.info(f"Training Arguments: {training_args}")
    logger.info(f"Remaining Arguments: {remaining_args}")
    set_seed(args.seed)

    if args.do_train and os.path.exists(args.output_dir):
        logger.warning(f"Output directory {args.output_dir} already exists!")
        if not args.overwrite_output_dir:
            logger.warning("Overwrite is disabled. Exiting...")
            return

    checkpoint_dir, completed_training = get_last_checkpoint(args.output_dir)
    if completed_training:
        logger.warning('Detected that training was already completed!')
        return

    tokenizer = get_tokenizer(args)
    logger.info('loaded tokenizer')

    with training_args.main_process_first(desc="Loading dataset"):
        data_module = make_data_module(tokenizer=tokenizer, args=args)
        logger.info('loaded data module')

    model = get_accelerate_model(args, checkpoint_dir)
    if not args.deepspeed:
        print_trainable_parameters(args, model)

    logger.info('loaded model')

    trainer = Seq2SeqTrainer(
    # trainer = PeftTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **{k:v for k,v in data_module.items() if k != 'predict_dataset'},
    )

    # Callbacks
    training_callbacks = [SavePeftModelCallback, CleanMemoryCallback, LoggingCallback]
    for callback in training_callbacks:
        trainer.add_callback(callback)

    # Verifying the datatypes.
    if not args.full_finetune:
        dtypes = {}
        for _, p in model.named_parameters():
            dtype = p.dtype
            if dtype not in dtypes: dtypes[dtype] = 0
            dtypes[dtype] += p.numel()
        total = 0
        for k, v in dtypes.items(): total+= v
        for k, v in dtypes.items():
            print(k, v, v/total)

    all_metrics = {"run_name": args.run_name}
    # Training
    if args.do_train:
        logger.info("*** Train ***")
        # Note: `resume_from_checkpoint` not supported for adapter checkpoints by HF.
        # Currently adapter checkpoint is reloaded as expected but optimizer/scheduler states are not.
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        all_metrics.update(metrics)
    # Evaluation
    if args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        all_metrics.update(metrics)
    # Prediction
    if args.do_predict:
        import numpy as np
        from speechless.finetune.dataset_utils import IGNORE_INDEX
        logger.info("*** Predict ***")
        prediction_output = trainer.predict(test_dataset=data_module['predict_dataset'],metric_key_prefix="predict")
        prediction_metrics = prediction_output.metrics
        predictions = prediction_output.predictions
        predictions = np.where(predictions != IGNORE_INDEX, predictions, tokenizer.pad_token_id)
        predictions = tokenizer.batch_decode(
            predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        with open(os.path.join(args.output_dir, 'predictions.jsonl'), 'w') as fout:
            for i, example in enumerate(data_module['predict_dataset']):
                example['prediction_with_input'] = predictions[i].strip()
                example['prediction'] = predictions[i].replace(example['input'], '').strip()
                fout.write(json.dumps(example) + '\n')
        print(prediction_metrics)
        trainer.log_metrics("predict", prediction_metrics)
        trainer.save_metrics("predict", prediction_metrics)
        all_metrics.update(prediction_metrics)

    if (args.do_train or args.do_eval or args.do_predict):
        with open(os.path.join(args.output_dir, "metrics.json"), "w") as fout:
            fout.write(json.dumps(all_metrics))

if __name__ == "__main__":
    finetune()
