import os
import transformers
from typing import Optional, Literal
from dataclasses import dataclass, field

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="EleutherAI/pythia-12b"
    )
    trust_remote_code: Optional[bool] = field(
        default=True,
        metadata={"help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."}
    )
    # use_auth_token: Optional[bool] = field(
    #     default=False,
    #     metadata={"help": "Enables using Huggingface auth token from Git Credentials."}
    # )

@dataclass
class DataArguments:

    mix_strategy: Literal["concat", "interleave_under", "interleave_over"] = field(
        default="concat",
        metadata={"help": "Strategy to use in dataset mixing (concat/interleave) (undersampling/oversampling)."},
    )
    streaming: bool = field(
        default=False,
        metadata={"help": "Enable dataset streaming."},
    )

    force_remove_overlength_samples: bool = field(
        default=True,
        metadata={"help": "Remove overlength samples."}
    )
    eval_dataset_size: float = field(
        default=0.02, metadata={"help": "Ratio of dataset to use for validation."}
    )
    max_train_samples: Optional[float] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set. If set to a float, will truncate the number of examples to that percentage of the dataset."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum model length (input and output).  Sequences will be right padded (and possibly truncated)."},
    )
    dataset: str = field(
        default='alpaca',
        metadata={"help": "Which dataset to finetune on. See datamodule for options."}
    )
    dataset_format: Optional[str] = field(
        default="conversations",
        metadata={"help": "Which dataset format is used. [alpaca|conversations|chip2|self-instruct|hh-rlhf|mistral]"}
    )
    prompt_type: Optional[str] = field(
        default=None,
        metadata={"help": "Which prompt type to use. [alpaca|chatlm|llama2|minicpm|conversations|chip2|self-instruct|hh-rlhf|mistral]"}
    )

@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    task_name: str = field(
        default=None,
        metadata={"help": "The name of the task to train on."},
    )
    flash_attention: bool = field(
        default=True,
        metadata={"help": "Use flash attention."}
    )

    long_lora: bool = field(
        default=False,
        metadata={"help": "Use long lora."}
    )

    rerope: bool = field(
        default=False,
        metadata={"help": "Use rerope."}
    )
    rerope_window: int = field(
        default=None,
        metadata={"help": "Rerope window size."}
    )

    neftune: bool = field(
        default=False,
        metadata={"help": "Use neftune."}
    )
    noise_alpha: float = field(
        default=5.0,
        metadata={"help": "Neftune noise alpha."}
    )


    sliding_window: int = field(
        default=4096,
        metadata={"help": "Sliding window size."}
    )

    rope_theta: float = field(
        default=10000,
        metadata={"help": "Rope theta."}
    )

    wandb: str = field(
        default=None,
        metadata={"help": "Wandb project name."}
    )

    sample_packing: bool = field(
        default=False,
        metadata={"help": "Use sample packing for effiecient training."}
    )
    cache_dir: Optional[str] = field(
        default=None
    )
    full_finetune: bool = field(
        default=False,
        metadata={"help": "Finetune the entire model without adapters."}
    )
    adam8bit: bool = field(
        default=False,
        metadata={"help": "Use 8-bit adam."}
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=4,
        metadata={"help": "How many bits to use."}
    )
    lora_r: int = field(
        default=64,
        metadata={"help": "Lora R dimension."}
    )
    lora_alpha: float = field(
        default=16,
        metadata={"help": " Lora alpha."}
    )
    lora_dropout: float = field(
        default=0.0,
        metadata={"help":"Lora dropout."}
    )
    max_memory_MB: int = field(
        default=80000,
        metadata={"help": "Free memory per gpu."}
    )
    output_dir: str = field(default='./output', metadata={"help": 'The output dir for logs and checkpoints'})
    optim: str = field(default='paged_adamw_32bit', metadata={"help": 'The optimizer to be used'})
    per_device_train_batch_size: int = field(default=1, metadata={"help": 'The training batch size per GPU. Increase for better speed.'})
    gradient_accumulation_steps: int = field(default=1, metadata={"help": 'How many gradients to accumulate before to perform an optimizer step'})
    num_train_epochs: int = field(default=3, metadata={"help": 'Number of training epochs.'})
    weight_decay: float = field(default=0.0, metadata={"help": 'The L2 weight decay rate of AdamW'}) # use lora dropout instead for regularization if needed
    learning_rate: float = field(default=0.0002, metadata={"help": 'The learning rate'})
    remove_unused_columns: bool = field(default=False, metadata={"help": 'Removed unused columns. Needed to make this codebase work.'})
    max_grad_norm: float = field(default=0.3, metadata={"help": 'Gradient clipping max norm. This is tuned and works well for all models tested.'})
    gradient_checkpointing: bool = field(default=False, metadata={"help": 'Use gradient checkpointing. You want to use this.'})
    do_train: bool = field(default=True, metadata={"help": 'To train or not to train, that is the question?'})
    lr_scheduler_type: str = field(default='constant', metadata={"help": 'Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis'})
    warmup_ratio: float = field(default=0.005, metadata={"help": 'Fraction of steps to do a warmup for'})
    logging_steps: int = field(default=10, metadata={"help": 'The frequency of update steps after which to log the loss'})
    group_by_length: bool = field(default=True, metadata={"help": 'Group sequences into batches with same length. Saves memory and speeds up training considerably.'})
    save_strategy: str = field(default='steps', metadata={"help": 'When to save checkpoints'})
    save_steps: int = field(default=250, metadata={"help": 'How often to save a model'})
    save_total_limit: int = field(default=1, metadata={"help": 'How many checkpoints to save before the oldest is overwritten'})
    deepspeed: str = field(default=None, metadata={"help": "deepspeed configuration path"})
    max_shard_size: str = field(default="5GB", metadata={"help": "Max shard size when saving model after full finetune."})

    def __post_init__(self):
        super().__post_init__()
        if self.run_name is None or self.run_name == self.output_dir:
            self.run_name = os.path.basename(os.curdir)

@dataclass
class GenerationArguments:
    # For more hyperparameters check:
    # https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
    # Length arguments
    max_new_tokens: Optional[int] = field(
        default=256,
        metadata={"help": "Maximum number of new tokens to be generated in evaluation or prediction loops"
                          "if predict_with_generate is set."}
    )
    min_new_tokens : Optional[int] = field(
        default=None,
        metadata={"help": "Minimum number of new tokens to generate."}
    )

    # Generation strategy
    do_sample: Optional[bool] = field(default=True)
    num_beams: Optional[int] = field(default=1)
    num_beam_groups: Optional[int] = field(default=1)
    penalty_alpha: Optional[float] = field(default=None)
    use_cache: Optional[bool] = field(default=True)

    # Hyperparameters for logit manipulation
    temperature: Optional[float] = field(default=1.0)
    top_k: Optional[int] = field(default=50)
    top_p: Optional[float] = field(default=1.0)
    typical_p: Optional[float] = field(default=1.0)
    diversity_penalty: Optional[float] = field(default=0.0)
    repetition_penalty: Optional[float] = field(default=1.0)
    length_penalty: Optional[float] = field(default=1.0)
    no_repeat_ngram_size: Optional[int] = field(default=0)

def get_args():
    import argparse
    hfparser = transformers.HfArgumentParser((
        ModelArguments, DataArguments, TrainingArguments 
    ))
    model_args, data_args, training_args, remaining_args = \
        hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )

    return args, model_args, data_args, training_args, remaining_args