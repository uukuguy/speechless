"""
Usage:

export MLX_MODEL_DIR=/opt/local/llm_models/huggingface.co/mlx-community && \
export MODEL_PATH=${MLX_MODEL_DIR}/Qwen2.5-3B-Instruct-bf16 && \
export PROMPT_FORMAT=chatml && \
export DATASET_PATH=/opt/local/datasets/OpenO1-SFT/OpenO1-SFT-instruction-response.jsonl && \
export EVAL_DATASET_SIZE=1000 && \
python -m speechless.finetune.mlx.sft \
    --model ${MODEL_PATH} \
    --prompt_format ${PROMPT_FORMAT} \
    --dataset_path ${DATASET_PATH}} \
    --eval_dataset_size ${EVAL_DATASET_SIZE} \
    --train-type lora-completion-only \
    --train_batch_size 4 \
    --eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --max_seq_length 2048 \
    --logging_steps 10 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_strategy epochs \
"""
import os, json
import warnings
from loguru import logger

import mlx.optimizers as optim
import numpy as np
from mlx_lm.tuner.trainer import TrainingArgs, default_loss, evaluate, train, iterate_batches
from mlx_lm.tuner.utils import linear_to_lora_layers
from mlx_lm.utils import load, generate, save_config
from mlx_lm.lora import print_trainable_parameters
from mlx_lm.tuner.datasets import Dataset as mlx_lm_dataset
from types import SimpleNamespace
import mlx.core as mx
from tqdm import tqdm
import mlx.nn as nn
import yaml
from speechless.finetune.mlx.tuning.utils import create_delineated_batches
# from speechless.finetune.mlx.dataset import Dataset
from speechless.finetune.mlx.config import CONFIG_DEFAULTS, yaml_loader, get_prompt_formatter, PROMPT_FORMATS
# from speechless.finetune.mlx.wandb import WandbCallback
# from mlx_tuning_fork.tuning.dynamic_learning import SCHEDULE_CONFIGURATION_TYPE_TO_CLASS
from pathlib import Path
from pprint import pprint


def completions_only_loss(model, inputs, input_lengths, lengths):
    shifted_inputs = inputs[:, :-1]
    shifted_labels = inputs[:, 1:]
    logits = model(shifted_inputs)
    logits = logits.astype(mx.float32)

    mask_width = shifted_inputs.shape[1]
    token_indices = mx.arange(mask_width)[None, :]
    mask = mx.logical_and(token_indices >= input_lengths[:, None], token_indices < lengths[:, None])

    ce = nn.losses.cross_entropy(logits, shifted_labels) * mask
    ntoks = mask.sum()
    ce = ce.sum() / ntoks
    return ce, ntoks


def completions_only_iterate_batches(dataset, tokenizer, batch_size, max_seq_length, train=False):
    idx = sorted(range(len(dataset)), key=lambda i: len(dataset[i]))
    if len(dataset) < batch_size:
        raise ValueError(
            f"Dataset must have at least batch_size={batch_size}"
            f" examples but only has {len(dataset)}."
        )

    # Make the batches:
    batch_idx = [
        idx[i: i + batch_size] for i in range(0, len(idx) - batch_size + 1, batch_size)
    ]
    while True:
        indices = np.random.permutation(len(batch_idx))
        for i in indices:
            input_text = []
            output_text = []

            for j in batch_idx[i]:
                record = dataset[j]
                input_text.append(prompt_formatter.get_input(record))
                output_text.append(prompt_formatter.get_output(record))
            yield create_delineated_batches(input_text, output_text, tokenizer, max_seq_length=max_seq_length)

        if not train:
            break


ALL_TRAIN_TYPES = ['lora-completion-only', 'dora-completion-only', 'lora-self-supervised',
                   'dora-self-supervised']
DORA_TRAIN_TYPES = ['dora-completion-only', 'dora-self-supervised']
COMPLETION_ONLY_TYPES = ['lora-completion-only', 'dora-completion-only']
PEFT_TYPES = ['lora-self-supervised', 'dora-self-supervised'] + COMPLETION_ONLY_TYPES


def get_args():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, required=True, help="Path to the model")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset, use comma to separate multiple datasets")
    parser.add_argument("--do_train", action="store_true", help="Whether to train the model")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Path to the output directory")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of train epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Warmup steps")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--num_layers", type=int, default=0, help="Number of layers to train, 0 for all layers")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=4, help="Evaluation batch size")
    parser.add_argument("--eval_dataset_size", type=float, default=200, help="Size of the evaluation dataset, 0.0 to 1.0 for proportion, otherwise number of records")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every X updates steps")
    parser.add_argument("--eval_steps", type=int, default=100, help="Evaluate every X updates steps")
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every X updates steps")
    parser.add_argument("--save_strategy", type=str, default="steps", choices=["steps", "epochs"], help="When to save checkpoints")
    parser.add_argument("--do_summary", action="store_true", help="Just summarize the training data")
    parser.add_argument("--verbose", action="store_true", help="Verbose mode")
    parser.add_argument("--train_type", type=str, default="lora-completion-only", choices=ALL_TRAIN_TYPES, help="Training type")
    parser.add_argument("--prompt_format", type=str, default="chatml", choices=PROMPT_FORMATS, help="Prompt format")
    parser.add_argument("--config_file", type=str, default="./lora_config.yaml", help="Path to the configuration file")

    args = parser.parse_args()
    return args

def main():
    import argparse
    param_args = get_args()
    logger.debug(param_args)

    tokenizer_config = {}

    param_dict = {}
    param_dict.update(CONFIG_DEFAULTS)

    if os.path.exists(param_args.config_file):
        with open(param_args.config_file, "r") as file:
            config = yaml.load(file, yaml_loader)
            param_dict.update(config)

    # if "model" not in param_dict:
    #     raise SyntaxError('Missing required "model" parameter')
    # for key, default in CONFIG_DEFAULTS.items():
    #     if key not in param_dict:
    #         param_dict[key] = default
    param_dict["verbose"] = param_args.verbose
    tokenizer_config = {"trust_remote_code": True if param_dict.get("trust_remote_code") else None}
    param_dict_eos_token = param_dict.get("eos_token")
    if param_dict_eos_token is not None:
        tokenizer_config["eos_token"] = param_dict["eos_token"]
    if param_args.verbose:
        pprint(param_dict)

    param_dict.update(vars(param_args))
    args = SimpleNamespace(**param_dict)

    logger.info(args)

    global prompt_formatter
    prompt_formatter = get_prompt_formatter(args.prompt_format)

    completion_only_training = args.train_type in COMPLETION_ONLY_TYPES

    print("Loading pretrained model")
    model, tokenizer = load(args.model, tokenizer_config=tokenizer_config)
    model.freeze()
    linear_to_lora_layers(
        model,
        args.num_layers,
        args.lora_parameters,
        use_dora=args.train_type in DORA_TRAIN_TYPES,
    )

    print_trainable_parameters(model)

    training_callback = None
    # if wandb_project:
    #     if wandb_run is None:
    #         raise RuntimeError("Specify the name of a Wandb run to use with --wandb-run ")
    #     try:
    #         import wandb
    #     except ImportError:
    #         raise ImportError('wandb module not available.  Install with `pip install wandb`')
    #     wandb.init(project=wandb_project, name=wandb_run, config=config)

    print("Loading datasets")

    if "," in args.dataset_path:
        dataset_paths = args.dataset_path.split(",")
    else:
        dataset_paths = [args.dataset_path]
    all_datasets = [[ json.loads(line) for line in open(dataset_path, 'r', encoding='utf-8').readlines()] for dataset_path in dataset_paths]
    all_datasets = np.concatenate(all_datasets)
    all_datasets = np.random.choice(all_datasets, len(all_datasets))

    if args.eval_dataset_size < 1.0:
        test_size = int(len(all_datasets) * args.eval_dataset_size)
    else:
        test_size = int(args.eval_dataset_size)
    valid_set = all_datasets[:test_size]
    train_set = all_datasets[test_size:]

    epoch_num_steps = (len(train_set) + args.train_batch_size - 1) // args.train_batch_size
    if args.num_train_epochs == -1:
        num_iterations = epoch_num_steps if args.iters == -1 else args.iters
    else:
        num_iterations = epoch_num_steps * args.num_train_epochs
    num_iterations = int(num_iterations)

    if args.save_strategy == "epochs":
        args.save_steps = epoch_num_steps

    args.train = args.do_train
    args.steps_per_report=args.logging_steps, #scaled_steps_per_report,
    args.steps_per_eval=args.eval_steps, # scaled_steps_per_eval,
    args.steps_per_save=args.save_steps, #scaled_save_every,

    # if wandb_project:
    #     training_callback = WandbCallback(tqdm(total=num_iterations))

    print(
        f"{num_iterations:,} iterations at {epoch_num_steps:,} iterations per epoch on a dataset of "
        f"{len(train_set):,} records, {args.train_batch_size} at a time and with a validation set of "
        f"{len(valid_set):,} records, training {args.num_layers} layers out of {len(model.layers)} using qLoRa."
    )

    # if args.evals_per_epoch:
    #     scaled_steps_per_eval = int(epoch_num_steps / args.evals_per_epoch)
    #     scaled_val_batches = int(len(valid_set) * args.eval_proportion_of_total / args.batch_size
    #                              ) if args.eval_proportion_of_total else (
    #         int(len(valid_set) / ((args.evals_per_epoch - 1) * args.batch_size))
    #     )
    # else:
    #     scaled_steps_per_eval = int(num_iterations * args.validation_interval_proportion)
    #     scaled_val_batches = int(args.validations_per_train_item * args.validation_interval_proportion * num_iterations)

    # scaled_steps_per_report = int(args.reporting_interval_proportion * num_iterations)

    # if args.saves_per_epoch:
    #     scaled_save_every = int(epoch_num_steps / args.saves_per_epoch)
    # else:
    #     scaled_save_every = int(args.adapter_save_interval_proportion * num_iterations)

    # print(
    #     f"Calculating loss every {scaled_steps_per_report:,} steps, reporting validation loss every "
    #     f"{scaled_steps_per_eval:,} steps, validating with {scaled_val_batches:,} batches, "
    #     f"and saving the adapter every {scaled_save_every:,} steps."
    # )

    iterate_batches_fn = completions_only_iterate_batches if completion_only_training else iterate_batches
    if not args.do_summary:

        # if "learning_schedule" in config:
        #     scheduler = SCHEDULE_CONFIGURATION_TYPE_TO_CLASS[
        #         config["learning_schedule"]["type"]].from_configuration(args.learning_rate, config, num_iterations)
        # else:
        #     scheduler = args.learning_rate
        scheduler = args.learning_rate


        # Resume training the given adapters.
        if args.resume_adapter_file is not None:
            print(f"Loading pretrained adapters from {args.resume_adapter_file}")
            model.load_weights(args.resume_adapter_file, strict=False)

        adapter_path = Path(args.adapter_path)
        adapter_path.mkdir(parents=True, exist_ok=True)
        save_config(vars(args), adapter_path / "adapter_config.json")
        adapter_file = adapter_path / "adapters.safetensors"

        training_args = TrainingArgs(
            batch_size=args.batch_size,
            iters=num_iterations,
            val_batches=-1, #scaled_val_batches,
            steps_per_report=args.logging_steps, #scaled_steps_per_report,
            steps_per_eval=args.eval_steps, # scaled_steps_per_eval,
            steps_per_save=args.save_steps, #scaled_save_every,
            adapter_file=adapter_file,
            max_seq_length=args.max_seq_length,
            grad_checkpoint=args.grad_checkpoint,
        )

        if args.do_train:
            print("Training")
            model.train()
            opt = optim.Adam(learning_rate=scheduler)
            train(
                model,
                tokenizer,
                opt,
                train_set,
                valid_set,
                args=training_args,
                loss=completions_only_loss if completion_only_training else default_loss,
                iterate_batches=iterate_batches_fn,
                training_callback=training_callback
            )

        # # Load the LoRA adapter weights which we assume should exist by this point
        # if not adapter_file.is_file():
        #     raise ValueError(
        #         f"Adapter file {adapter_file} missing. "
        #         "Use --train to learn and save the adapters"
        #     )
        # model.load_weights(str(adapter_file), strict=False)
        # print(f"Loaded weights from {adapter_file}")

        # if args.test:
        #     print(f"Testing ({len(test_set):,} records)")
        #     model.eval()

        #     test_loss = evaluate(
        #         model=model,
        #         dataset=test_set,
        #         tokenizer=tokenizer,
        #         batch_size=args.batch_size,
        #         num_batches=args.test_batches,
        #         loss=completions_only_loss,
        #         iterate_batches=completions_only_iterate_batches
        #     )

        #     test_ppl = math.exp(test_loss)

        #     print(f"Test loss {test_loss:.3f}, Test ppl {test_ppl:.3f}.")

    else:
        total_num_tokens = 0
        max_tokens = 0
        _lengths = []
        for it, (batch, input_lengths, lengths) in zip(
                range(1, num_iterations + 1),
                iterate_batches_fn(
                    dataset=train_set,
                    tokenizer=tokenizer,
                    batch_size=args.batch_size,
                    max_seq_length=args.max_seq_length,
                    train=False,
                )
        ):
            max_tokens = max(max_tokens, max(lengths))
            _lengths.extend(lengths)
            total_num_tokens += sum(lengths)
        print(f"A total of {total_num_tokens:,} training tokens, {total_num_tokens / num_iterations:.3f} per "
              f"step/iteration, an average of {total_num_tokens / len(_lengths):.3f} tokens per record, with"
              f" the largest having {max_tokens:,} tokens.")
        print(f"mlx_lm.lora --val-batches -1 \\\n"
              f"            --steps-per-report {args.logging_steps} \\\n"
              f"            --steps-per-eval {args.eval_steps} \\\n"
              f"            --save-every {args.save_steps} \\\n"
              f"            --iters {num_iterations} -c {args.config_file}")

if __name__ == '__main__':
    main()
