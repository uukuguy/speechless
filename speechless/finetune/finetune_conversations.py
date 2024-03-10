#!/usr/bin/env python
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import math
import gc, ctypes
#from rich import print

# if os.environ.get('ENABLE_FLASH_ATTENTION', 'False') == 'True':
#     from flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
#     replace_llama_attn_with_flash_attn(packed=True)
#     print(f"Enabled flash attention monkey patching.")

# if os.environ.get('ENABLE_REROPE', "False") == 'True':
#     from rerope_monkey_patch import replace_llama_attention_forword_with_rerope
#     replace_llama_attention_forword_with_rerope(training_length=4096, window=1536)
#     print(f"Enabled rerope monkey patching.")

import os, json
from os.path import exists, join, isdir
from dataclasses import dataclass, field
from typing import Optional, Dict 
import numpy as np
import bitsandbytes as bnb

# logger = logging.getLogger(__name__)
from loguru import logger

# from speechless.patches.llama_attn_replace_sft import replace_llama_attn
# replace_llama_attn(use_flash_attn=True, use_full=False, inference=False)

import torch
import transformers
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
    Seq2SeqTrainer,
    BitsAndBytesConfig,
)

from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from speechless.finetune.dataset_utils import IGNORE_INDEX
from speechless.finetune.dataset_utils import make_data_module


def qwen_prepare_model_for_kbit_training(model, use_gradient_checkpointing=True):
    """
    This method wraps the entire protocol for preparing a model before running a training. This includes:
        1- Cast the layernorm in fp32 2- making output embedding layer require grads 3- Add the upcasting of the lm
        head to fp32

    Args:
        model, (`transformers.PreTrainedModel`):
            The loaded model from `transformers`
    """
    loaded_in_kbit = getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False)
    is_gptq_quantized = getattr(model, "quantization_method", None) == "gptq"
    for name, param in model.named_parameters():
        # freeze base model's layers
        param.requires_grad = False

    # if not is_gptq_quantized:
    #     # cast all non INT8 parameters to fp32
    #     for param in model.parameters():
    #         if (param.dtype == torch.float16) or (param.dtype == torch.bfloat16):
    #             param.data = param.data.to(torch.float32)

    if (loaded_in_kbit or is_gptq_quantized) and use_gradient_checkpointing:
        # For backward compatibility
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        # enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()

    return model



torch.backends.cuda.matmul.allow_tf32 = True

def clean_memory():
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)
    torch.cuda.empty_cache()

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
    # report_to: str = field(
    #     default='none',
    #     metadata={"help": "To use wandb or something else for reporting."}
    # )
    mpt: bool = field(default=False, metadata={"help": 'Flag indicating whether this model is MPT or not'})
    output_dir: str = field(default='./output', metadata={"help": 'The output dir for logs and checkpoints'})
    optim: str = field(default='paged_adamw_32bit', metadata={"help": 'The optimizer to be used'})
    per_device_train_batch_size: int = field(default=1, metadata={"help": 'The training batch size per GPU. Increase for better speed.'})
    gradient_accumulation_steps: int = field(default=16, metadata={"help": 'How many gradients to accumulate before to perform an optimizer step'})
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

    repeat_steps: int = field(default=0, metadata={"help": "How many times to repeat the same batch."})

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
    do_sample: Optional[bool] = field(default=False)
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

def find_all_linear_names(args, model):
    cls = bnb.nn.Linear4bit if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

class CleanMemoryCallback(transformers.TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        clean_memory()

    def on_evaluate(self, args, state, control, **kwargs):
        clean_memory()

from transformers import TrainerCallback
class LoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            if "eval_loss" in logs:
                logger.info(logs)
            else:
                logger.debug(logs)

class SavePeftModelCallback(TrainerCallback):
    def save_model(self, args, state, kwargs):
        logger.info('Saving PEFT checkpoint...')
        # if state.best_model_checkpoint is not None:
        #     checkpoint_folder = os.path.join(state.best_model_checkpoint, "adapter_model")
        # else:
        #     checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)

        self._symlink_latest_checkpoint(checkpoint_folder)

    def _symlink_latest_checkpoint(self, checkpoint_folder):
        # if the latest checkpoint is a symlink, remove it
        output_dir = os.path.dirname(checkpoint_folder)
        latest_checkpoint = os.path.join(output_dir, "latest")
        if os.path.islink(latest_checkpoint):
            os.remove(latest_checkpoint)
        # symlink the latest checkpoint to the checkpoint folder
        os.symlink(os.path.basename(checkpoint_folder), latest_checkpoint)

    def on_save(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        def touch(fname, times=None):
            with open(fname, 'a'):
                os.utime(fname, times)

        if state.is_local_process_zero:
            touch(join(args.output_dir, 'completed'))
            self.save_model(args, state, kwargs)

def get_accelerate_model(args, checkpoint_dir):

    n_gpus = torch.cuda.device_count()
    max_memory = f'{args.max_memory_MB}MB'
    max_memory = {i: max_memory for i in range(n_gpus)}
    device_map = "auto"

    # if we are in a distributed setting, we need to set the device map and max memory per device
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': f'cuda:{local_rank}'}
        max_memory = {'': max_memory[local_rank]}


    if args.full_finetune: assert args.bits in [16, 32]

    logger.info(f'loading base model {args.model_name_or_path}...')

    config = transformers.AutoConfig.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
    )
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    config.sliding_window = args.sliding_window
    config.rope_theta = args.rope_theta


    if "qwen" in args.model_name_or_path.lower():
        if args.bf16:
            config.bf16 = True
        elif args.fp16:
            config.fp16 = True

    compute_dtype = (torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))
    model_kwargs = {
        "cache_dir": args.cache_dir,
        # transformers-4.39.0.dev0
        # ValueError: You can't pass `load_in_4bit`or `load_in_8bit` as a kwarg when passing `quantization_config` argument at the same time.
        # "load_in_4bit": args.bits == 4,
        # "load_in_8bit": args.bits == 8,
        "device_map": device_map if not args.deepspeed else None,
        "max_memory": max_memory if not args.deepspeed else None,
        "quantization_config": BitsAndBytesConfig(
            load_in_4bit=args.bits == 4,
            load_in_8bit=args.bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.double_quant,
            bnb_4bit_quant_type=args.quant_type,
        ) if args.bits in (4, 8) else None,
        "torch_dtype": (torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)),
        "trust_remote_code": args.trust_remote_code,
        "use_flash_attention_2": args.flash_attention,
        # "use_auth_token": args.use_auth_token
    }

    # if args.mpt:
    #     model_kwargs["attn_config"] = {"attn_impl": "triton"}

    logger.info(f"{model_kwargs=}")
    # model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, config=config, **model_kwargs)
    if compute_dtype == torch.float16 and args.bits == 4:
        major, minor = torch.cuda.get_device_capability()
        if major >= 8:
            print('='*80)
            print('Your GPU supports bfloat16, you can accelerate training with the argument --bf16')
            print('='*80)

    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)

    model.config.torch_dtype=(torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))

    if not args.full_finetune:

        if "qwen" in args.model_name_or_path.lower():
            model = qwen_prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
        else:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if not args.full_finetune:
        if checkpoint_dir is not None:
            logger.info("Loading adapters from checkpoint.")
            model = PeftModel.from_pretrained(model, join(checkpoint_dir, 'adapter_model'), is_trainable=True)
        else:
            logger.info(f'adding LoRA modules...')
            modules = find_all_linear_names(args, model)
            print(f"LoRA modules: {modules}")
            config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=modules,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, config)

    for name, module in model.named_modules():
        if "norm" in name:
            module.to(compute_dtype)
        if "lm_head" in name or "embed_tokens" in name:
            if hasattr(module, "weight"):
                module.to(compute_dtype)

    if args.neftune:
        from speechless.patches.neftune_monkey_patch import NEFTune
        model = NEFTune(model, noise_alpha=args.noise_alpha)

    return model

def print_trainable_parameters(args, model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    if args.bits == 4: trainable_params /= 2
    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable: {100 * trainable_params / all_param}"
    )

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

#FIXME

def get_last_checkpoint(checkpoint_dir):
    if isdir(checkpoint_dir):
        is_completed = exists(join(checkpoint_dir, 'completed'))
        if is_completed: return None, True # already finished
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if isdir(join(checkpoint_dir, filename)) and filename.startswith('checkpoint'):
                max_step = max(max_step, int(filename.replace('checkpoint-', '')))
        if max_step == 0: return None, is_completed # training started, but no checkpoint
        checkpoint_dir = join(checkpoint_dir, f'checkpoint-{max_step}')
        logger.info(f"Found a previous checkpoint at: {checkpoint_dir}")
        return checkpoint_dir, is_completed # checkpoint found!
    return None, False # first training

# from peft.utils.other import WEIGHTS_NAME
# from transformers.trainer import TRAINING_ARGS_NAME
# from transformers.modeling_utils import unwrap_model

# def get_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]: # get state dict containing trainable parameters
#     state_dict = model.state_dict()
#     filtered_state_dict = {}

#     for k, v in model.named_parameters():
#         if v.requires_grad:
#             filtered_state_dict[k] = state_dict[k].cpu().clone().detach()

#     return filtered_state_dict

# class PeftTrainer(Seq2SeqTrainer):
#     r"""
#     Inherits Seq2SeqTrainer to support parameter-efficient checkpoints.
#     """

#     # def __init__(self, finetuning_args: FinetuningArguments, **kwargs):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         if self.is_world_process_zero() and os.path.exists(os.path.join(self.args.output_dir, "trainer_log.jsonl")):
#             logger.warning("Previous log file in this folder will be deleted.")
#             os.remove(os.path.join(self.args.output_dir, "trainer_log.jsonl"))

#     def _save(self, output_dir: Optional[str] = None, state_dict: Optional[Dict[str, torch.Tensor]] = None) -> None:
#         r"""
#         Saves trainable parameters as model checkpoint.

#         This function will only be executed at the process zero.

#         Subclass and override to inject custom behavior. It should not be directly used by external scripts.
#         """
#         output_dir = output_dir if output_dir is not None else self.args.output_dir
#         os.makedirs(output_dir, exist_ok=True)
#         logger.info(f"Saving model checkpoint to {output_dir}")
#         model = unwrap_model(self.model)

#         if hasattr(model, "pretrained_model"): # for models with valuehead
#             backbone_model = getattr(model, "pretrained_model")
#         else:
#             backbone_model = model

#         if hasattr(backbone_model, "peft_config"): # peft methods
#             backbone_model.save_pretrained(output_dir, state_dict=get_state_dict(backbone_model)) # save lora weights
#         else:
#             torch.save(get_state_dict(backbone_model), os.path.join(output_dir, WEIGHTS_NAME)) # save trainable weights

#         # if hasattr(model, "v_head"): # save valuehead weights
#         #     torch.save(get_state_dict(getattr(model, "v_head")), os.path.join(output_dir, VALUE_HEAD_FILE_NAME))

#         with open(os.path.join(output_dir, TRAINING_ARGS_NAME), "w", encoding="utf-8") as f:
#             f.write(self.args.to_json_string() + "\n")
#         # self.finetuning_args.save_to_json(os.path.join(output_dir, FINETUNING_ARGS_NAME))

#     # def _load_best_model(self):
#     #     r"""
#     #     Loads trainable parameters from model checkpoint.

#     #     Subclass and override to inject custom behavior. It should not be directly used by external scripts.
#     #     """
#     #     logger.info(f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric}).")
#     #     model = unwrap_model(self.model)
#     #     if hasattr(model, "peft_config"): # peft methods
#     #         model.load_adapter(self.state.best_model_checkpoint, getattr(model, "active_adapter"))
#     #     else:
#     #         load_trainable_params(model, self.state.best_model_checkpoint)

#     #     if hasattr(model, "v_head"):
#     #         load_valuehead_params(model, self.state.best_model_checkpoint)

def setup_wandb(args):
    # Check if parameter passed or if set within environ
    wandb_project = args.wandb
    if wandb_project is None:
        return

    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    # if len(wandb_watch) > 0:
    #     os.environ["WANDB_WATCH"] = wandb_watch
    # if len(wandb_log_model) > 0:
    #     os.environ["WANDB_LOG_MODEL"] = wandb_log_model

from transformers import SchedulerType
from transformers.optimization import TYPE_TO_SCHEDULER_FUNCTION
def get_lr_scheduler(lr_scheduler_type, optimizer, num_warmpup_steps, num_training_steps, optim_args):
    name = SchedulerType(lr_scheduler_type)
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]
    optim_args = optim_args or {}
    return schedule_func(optimizer, num_warmup_steps=num_warmpup_steps, num_training_steps=num_training_steps, **optim_args)

def train():
    hfparser = transformers.HfArgumentParser((
        ModelArguments, DataArguments, TrainingArguments, GenerationArguments
    ))
    model_args, data_args, training_args, generation_args, extra_args = \
        hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    # training_args.generation_config = transformers.GenerationConfig(**vars(generation_args))
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )

    if args.task_name is None:
        args.task_name = os.path.basename(os.curdir)

    print(f"{args=}")

    from datetime import datetime
    logger.add(f"{args.output_dir}/logs/{args.task_name}-{datetime.now().strftime('%Y%m%d_%H%M%S')}.log", level="INFO")
    logger.info(f"{args=}")

    # setup_wandb(args)

    # if args.flash_attention:
    #     from speechless.patches.flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
    #     replace_llama_attn_with_flash_attn(packed=args.sample_packing)
    #     logger.info(f"Enabled flash attention monkey patching.")

    if args.long_lora:
        if 'mistral' in args.model_name_or_path:
            logger.warning(f"Mistral doesn't support long alpaca now.")
            # from speechless.patches.llama_attn_replace_sft import replace_mistral_attn
            # replace_mistral_attn(use_flash_attn=training_args.flash_attention, use_full=False, inference=False)
        else:
            from speechless.patches.long_lora_monkey_patch import replace_llama_attn
            replace_llama_attn(use_flash_attn=training_args.flash_attention, use_full=False, inference=False)

    if args.rerope:
        from speechless.patches.rerope_monkey_patch import replace_llama_attention_forword_with_rerope
        rerope_window = args.rerope_window or int(args.model_max_length * 3 / 8) // 16 * 16
        replace_llama_attention_forword_with_rerope(training_length=args.model_max_length, window=rerope_window)
        logger.info(f"Enabled rerope monkey patching.")

    # if args.sliding_window > 0:
    #     if 'mistral' not in args.model_name_or_path:
    #         from speechless.patches.sliding_window_monkey_patch import replace_llama_attn
    #         replace_llama_attn()

    checkpoint_dir, completed_training = get_last_checkpoint(args.output_dir)
    if completed_training:
        logger.warning('Detected that training was already completed!')

    model = get_accelerate_model(args, checkpoint_dir)

    model.config.use_cache = False
    if not args.deepspeed:
        print_trainable_parameters(args, model)
    logger.info('loaded model')
    set_seed(args.seed)

    # Tokenizer
    tokenizer_kwargs = {
        "cache_dir": args.cache_dir,
        "padding_side": "left",
        "use_fast": False,
        "trust_remote_code": True,
    }
    # if args.mpt:
    #     tokenizer_kwargs["padding_side"] = "left"
    #     tokenizer_kwargs.pop("use_fast")
    print(f"---------- Original tokens----------")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, **tokenizer_kwargs)
    print(f"{tokenizer.pad_token=},{tokenizer.pad_token_id=}")
    print(f"{tokenizer.unk_token=},{tokenizer.unk_token_id=}")
    print(f"{tokenizer.bos_token=},{tokenizer.bos_token_id=}")
    print(f"{tokenizer.eos_token=},{tokenizer.eos_token_id=}")

    if "qwen" in args.model_name_or_path.lower():
        tokenizer.eos_token = "<|endoftext|>"
        # tokenizer.unk_token = "<|extra_3|>"
        tokenizer.bos_token = "<|extra_2|>"
        tokenizer.pad_token = "<|extra_1|>"
    else:
        if tokenizer.bos_token_id is None:
            tokenizer.bos_token_id = 1
            tokenizer.bos_token = "<s>"
        if tokenizer.eos_token_id is None:
            tokenizer.eos_token_id = 2
            tokenizer.eos_token = "</s>"
        # if tokenizer.unk_token_id is None:
        #     tokenizer.unk_token_id = 0
        #     tokenizer.unk_token = "<unk>"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = 0 # tokenizer.eos_token_id
            tokenizer.pad_token = tokenizer._convert_id_to_token(tokenizer.pad_token_id) #tokenizer.eos_token
    print(f"---------- Fixed tokens ----------")
    print(f"{tokenizer.pad_token=},{tokenizer.pad_token_id=}")
    # print(f"{tokenizer.unk_token=},{tokenizer.unk_token_id=}")
    print(f"{tokenizer.bos_token=},{tokenizer.bos_token_id=}")
    print(f"{tokenizer.eos_token=},{tokenizer.eos_token_id=}")
    data_module = make_data_module(tokenizer=tokenizer, args=args)
    trainer = Seq2SeqTrainer(
    # trainer = PeftTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **{k:v for k,v in data_module.items() if k != 'predict_dataset'},
    )

    # Callbacks
    if not args.full_finetune:
        trainer.add_callback(SavePeftModelCallback)

    trainer.add_callback(CleanMemoryCallback)
    trainer.add_callback(LoggingCallback)


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

    # Safely save final full-tune model.
    if args.full_finetune:
        state_dict = trainer.model.state_dict()
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        trainer.model.save_pretrained(args.output_dir, state_dict=cpu_state_dict, max_shard_size=args.max_shard_size)
        tokenizer.save_pretrained(args.output_dir)
        with open(os.path.join(args.output_dir, "config.json")) as infile:
            config = json.loads(infile.read())
        config["_name_or_path"] = os.path.basename(args.output_dir)
        with open(os.path.join(args.output_dir, "config.json"), "w") as outfile:
            outfile.write(json.dumps(config, indent=2))


if __name__ == "__main__":
    train()
