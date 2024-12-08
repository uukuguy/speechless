"""
Usage:
python -m speechless.finetune.mlx.sft \
    --model 
    --train-type lora-completion-only
"""
import warnings

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
import click
import yaml
import math
from speechless.finetune.mlx.tuning.utils import create_delineated_batches
from speechless.finetune.mlx.dataset import Dataset
from speechless.finetune.mlx.config import CONFIG_DEFAULTS, yaml_loader, get_prompt_formatter, PROMPT_FORMATS
from speechless.finetune.mlx.wandb import WandbCallback
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

@click.command()
@click.option('--verbose/--no-verbose', default=False)
@click.option("--summary/--no-summary", default=False, help="Just summarize training data")
@click.option('--train-type',
              type=click.Choice(ALL_TRAIN_TYPES, case_sensitive=False),
              default="lora-completion-only")
@click.option('-f', '--prompt-format',
              type=click.Choice(PROMPT_FORMATS, case_sensitive=False))
@click.option('--wandb-project', default=None, type=str,
              help='Wandb project name')
@click.option('--wandb-run', default=None, type=str,
              help='Wandb run name')
@click.argument('config_file')
def main(verbose, summary, train_type, prompt_format, wandb_project, wandb_run, config_file):
    global prompt_formatter
    prompt_formatter = get_prompt_formatter(prompt_format)
    tokenizer_config = {}
    with open(config_file, "r") as file:
        config = yaml.load(file, yaml_loader)
        param_dict = {k: v for k, v in config.items()}
        if "model" not in param_dict:
            raise SyntaxError('Missing required "model" parameter')
        for key, default in CONFIG_DEFAULTS.items():
            if key not in param_dict:
                param_dict[key] = default
        param_dict["verbose"] = verbose
        tokenizer_config = {"trust_remote_code": True if param_dict.get("trust_remote_code") else None}
        param_dict_eos_token = param_dict.get("eos_token")
        if param_dict_eos_token is not None:
            tokenizer_config["eos_token"] = param_dict["eos_token"]
        if verbose:
            pprint(param_dict)
        args = SimpleNamespace(**param_dict)

    completion_only_training = train_type in COMPLETION_ONLY_TYPES

    print("Loading pretrained model")
    model, tokenizer = load(args.model, tokenizer_config=tokenizer_config)
    model.freeze()
    linear_to_lora_layers(
        model,
        args.num_layers,
        args.lora_parameters,
        use_dora=train_type in DORA_TRAIN_TYPES,
    )

    print_trainable_parameters(model)

    training_callback = None
    if wandb_project:
        if wandb_run is None:
            raise RuntimeError("Specify the name of a Wandb run to use with --wandb-run ")
        try:
            import wandb
        except ImportError:
            raise ImportError('wandb module not available.  Install with `pip install wandb`')
        wandb.init(project=wandb_project, name=wandb_run, config=config)

    print("Loading datasets")
    names = ("train", "valid", "test")
    if train_type in ('lora-completion-only', 'dora-completion-only', 'debug'):
        train_set, valid_set, test_set = (Dataset(Path(args.data) / f"{n}.jsonl") for n in names)
    else:
        train_set, valid_set, test_set = (mlx_lm_dataset(Path(args.data) / f"{n}.jsonl") for n in names)

    if args.train and len(train_set) == 0:
        raise ValueError(
            "Training set not found or empty. Must provide training set for fine-tuning."
        )
    if args.train and len(valid_set) == 0:
        warnings.warn(
            "Validation set not found or empty. Should provide validation set for fine-tuning."
        )
    if args.test and len(test_set) == 0:
        raise ValueError(
            "Test set not found or empty. Must provide test_set set for evaluation."
        )

    epoch_num_steps = (len(train_set) + args.batch_size - 1) // args.batch_size
    if args.epochs == -1:
        num_iterations = epoch_num_steps if args.iters == -1 else args.iters
    else:
        num_iterations = epoch_num_steps * args.epochs
    num_iterations = int(num_iterations)

    if wandb_project:
        training_callback = WandbCallback(tqdm(total=num_iterations))

    print(
        f"{num_iterations:,} iterations at {epoch_num_steps:,} iterations per epoch on a dataset of "
        f"{len(train_set):,} records, {args.batch_size} at a time and with a validation set of "
        f"{len(valid_set):,} records, training {args.num_layers} layers out of {len(model.layers)} using qLoRa."
    )

    if args.evals_per_epoch:
        scaled_steps_per_eval = int(epoch_num_steps / args.evals_per_epoch)
        scaled_val_batches = int(len(valid_set) * args.eval_proportion_of_total / args.batch_size
                                 ) if args.eval_proportion_of_total else (
            int(len(valid_set) / ((args.evals_per_epoch - 1) * args.batch_size))
        )
    else:
        scaled_steps_per_eval = int(num_iterations * args.validation_interval_proportion)
        scaled_val_batches = int(args.validations_per_train_item * args.validation_interval_proportion * num_iterations)

    scaled_steps_per_report = int(args.reporting_interval_proportion * num_iterations)

    if args.saves_per_epoch:
        scaled_save_every = int(epoch_num_steps / args.saves_per_epoch)
    else:
        scaled_save_every = int(args.adapter_save_interval_proportion * num_iterations)

    print(
        f"Calculating loss every {scaled_steps_per_report:,} steps, reporting validation loss every "
        f"{scaled_steps_per_eval:,} steps, validating with {scaled_val_batches:,} batches, "
        f"and saving the adapter every {scaled_save_every:,} steps."
    )

    iterate_batches_fn = completions_only_iterate_batches if completion_only_training else iterate_batches
    if not summary:

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
            val_batches=scaled_val_batches,
            steps_per_report=scaled_steps_per_report,
            steps_per_eval=scaled_steps_per_eval,
            steps_per_save=scaled_save_every,
            adapter_file=adapter_file,
            max_seq_length=args.max_seq_length,
            grad_checkpoint=args.grad_checkpoint,
        )

        if args.train:
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

        # Load the LoRA adapter weights which we assume should exist by this point
        if not adapter_file.is_file():
            raise ValueError(
                f"Adapter file {adapter_file} missing. "
                "Use --train to learn and save the adapters"
            )
        model.load_weights(str(adapter_file), strict=False)
        print(f"Loaded weights from {adapter_file}")

        if args.test:
            print(f"Testing ({len(test_set):,} records)")
            model.eval()

            test_loss = evaluate(
                model=model,
                dataset=test_set,
                tokenizer=tokenizer,
                batch_size=args.batch_size,
                num_batches=args.test_batches,
                loss=completions_only_loss,
                iterate_batches=completions_only_iterate_batches
            )

            test_ppl = math.exp(test_loss)

            print(f"Test loss {test_loss:.3f}, Test ppl {test_ppl:.3f}.")

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
        print(f"mlx_lm.lora --val-batches {scaled_val_batches} \\\n"
              f"            --steps-per-report {scaled_steps_per_report} \\\n"
              f"            --steps-per-eval {scaled_steps_per_eval} \\\n"
              f"            --save-every {scaled_save_every} \\\n"
              f"            --iters {num_iterations} -c {config_file}")

if __name__ == '__main__':
    main()
