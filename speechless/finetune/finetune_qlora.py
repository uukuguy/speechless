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
from transformers import set_seed, Seq2SeqTrainer, PreTrainedTokenizer
from typing import Dict
import bitsandbytes as bnb
from loguru import logger

from speechless.utils.model_utils import get_accelerate_model, print_trainable_parameters
from speechless.utils.tokenizer_utils import get_tokenizer
from speechless.finetune.dataset_utils import make_data_module
from speechless.finetune.callbacks import LoggingCallback, CleanMemoryCallback, SavePeftModelCallback
from speechless.finetune.finetune_qlora_arguments import get_args

import pandas as pd
from datasets import load_dataset, Dataset, concatenate_datasets
from speechless.finetune.dataset_utils.data_collectors import DialogDataCollatorForCausalLM
def make_data_module(tokenizer: PreTrainedTokenizer, args) -> Dict:

    def local_dataset(dataset_name, test_size=0.02, stratify_by_column=None):
        if "," in dataset_name:
            all_datasets = []
            for dataset_attr in [ n.strip() for n in dataset_name.split(',')]:
                all_datasets.append(load_dataset("json", data_files=dataset_attr))
            full_dataset = concatenate_datasets(all_datasets)
        else:
            if dataset_name.endswith(('.json', '.jsonl')):
                full_dataset = Dataset.from_json(path_or_paths=dataset_name)
            elif dataset_name.endswith('.csv'):
                full_dataset = Dataset.from_pandas(pd.read_csv(dataset_name))
            elif dataset_name.endswith('.tsv'):
                full_dataset = Dataset.from_pandas(pd.read_csv(dataset_name, delimiter='\t'))
            else:
                raise ValueError(f"Unsupported dataset format: {dataset_name}")

        if test_size < 1.0:
            test_size = int(test_size * len(full_dataset))
        else:
            test_size = int(test_size)
        if stratify_by_column is not None and stratify_by_column in full_dataset.column_names:
            full_dataset = full_dataset.class_encode_column(stratify_by_column)

        return full_dataset.train_test_split(test_size=test_size, stratify_by_column=stratify_by_column)


    def load_data(dataset_name):
        if os.path.exists(dataset_name):
            try:
                full_dataset = local_dataset(dataset_name, test_size=args.eval_dataset_size)
                return full_dataset
            except:
                raise ValueError(f"Error loading dataset from {dataset_name}")
        else:
            raise NotImplementedError(f"Dataset {dataset_name} not implemented yet.")

    def format_dataset(dataset, dataset_format):
        assert dataset_format in ['instruction-response', 'conversations']

        if dataset_format == 'conversations':

            def _format_multi_turns(example):
                human_bot_dialog = []
                dialog = example['conversations']
                for round in dialog:
                    who = round['from']
                    response = round['value']
                    human_bot_dialog.append({
                        "from": who,
                        "value": response,
                    })
                return {
                    'conversations': human_bot_dialog
                }

            dataset = dataset.map(_format_multi_turns)
        elif dataset_format == 'instruction-response':
            def _format_input_output(example):
                return {
                    'conversations': [
                        {
                            "from": "human", 
                            "value": example['instruction'] 
                        },
                        {
                            "from": "assistant", 
                            "value": example['response'] 
                        },
                        ]
                }

            dataset = dataset.map(_format_input_output)

        # Remove unused columns.
        dataset = dataset.remove_columns(
            # FIXME
            [
                col for col in dataset.column_names['train']
                if col not in ['conversations', 'system_prompt', 'prompt_type']
            ]
        )
        return dataset


    # Load dataset.
    dataset = load_data(args.dataset)

    args.dataset_format = args.dataset_format if args.dataset_format else "instruction-response"
    dataset = format_dataset(dataset, args.dataset_format)

    logger.info(f"---------- Splitting dataset into train/eval ----------")
    if args.do_eval or args.do_predict:
        if 'eval' in dataset:
            eval_dataset = dataset['eval']
        elif 'test' in dataset:
            eval_dataset = dataset['test']
        else:
            logger.info('Splitting train dataset in train and validation according to `eval_dataset_size`')
            if 'category' in dataset["train"].column_names:
                dataset["train"] = dataset["train"].class_encode_column('category')
                dataset = dataset["train"].train_test_split(
                    test_size=args.eval_dataset_size, stratify_by_column='category', seed=args.seed
                )
            else:
                dataset = dataset["train"].train_test_split(
                    test_size=args.eval_dataset_size, shuffle=True, seed=args.seed
                )
            eval_dataset = dataset['test']
        if args.max_eval_samples is not None and len(eval_dataset) > args.max_eval_samples:
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))
        if args.group_by_length:
            eval_dataset = eval_dataset.map(lambda x: {'length': len(x['input']) + len(x['output'])})
    if args.do_train:
        train_dataset = dataset['train']
        max_train_samples = args.max_train_samples
        if max_train_samples is not None and max_train_samples > 0 and max_train_samples < 1.0:
            max_train_samples = int(len(train_dataset) * max_train_samples)
        else:
            max_train_samples = 0.0
        if max_train_samples >= 1.0 and len(train_dataset) > max_train_samples:
            train_dataset = train_dataset.select(range(max_train_samples))

        if args.group_by_length:
            train_dataset = train_dataset.map(lambda x: {'length': len(x['input']) + len(x['output'])})

    logger.info(f"{len(train_dataset)} training samples, {len(eval_dataset)} evaluation samples.")

    data_collator = DialogDataCollatorForCausalLM(
        tokenizer=tokenizer,
        model_max_length=args.model_max_length,
        prompt_type=args.prompt_type,
    )

    return dict(
        train_dataset=train_dataset if args.do_train else None,
        eval_dataset=eval_dataset if args.do_eval else None,
        predict_dataset=eval_dataset if args.do_predict else None,
        data_collator=data_collator
    )


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
