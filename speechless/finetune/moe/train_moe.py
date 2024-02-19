#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import os
from os.path import exists, join, isdir
import gc, ctypes
import math
from dataclasses import dataclass, field
from typing import Optional
from loguru import logger

import torch
from torch import nn
from torch.utils.data import Dataset
# import utils
import bitsandbytes as bnb

import transformers
from transformers import Trainer, BitsAndBytesConfig, set_seed
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft.tuners.lora import LoraLayer

from speechless.finetune.moe.camelidae.configuration_camelidae import CamelidaeConfig
from speechless.finetune.moe.camelidae.modeling_camelidae import LlamaForCausalLM

from speechless.finetune.moe.sparsetral.configuration_sparsetral import SparsetralConfig
from speechless.finetune.moe.sparsetral.modeling_sparsetral import MistralForCausalLM

import warnings

warnings.filterwarnings("ignore")

from speechless.finetune.moe.transformers_utils import (
    get_keys_to_not_convert,
    _load_pretrained_model,
)
import transformers.integrations
import transformers.modeling_utils

from speechless.finetune.dataset_utils import make_data_module

transformers.integrations.get_keys_to_not_convert = get_keys_to_not_convert
transformers.modeling_utils.PreTrainedModel._load_pretrained_model = (_load_pretrained_model)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    # data_path: str = field(
    #     default=None, metadata={"help": "Path to the training data."}
    # )
    force_remove_overlength_samples: bool = field(default=True, metadata={"help": "Remove overlength samples."})
    # group_by_length: bool = field(default=True, metadata={"help": 'Group sequences into batches with same length. Saves memory and speeds up training considerably.'})
    eval_dataset_size: float = field(default=0.02, metadata={"help": "Ratio of dataset to use for validation."})
    max_train_samples: Optional[float] = field(
        default=None,
        metadata={
            "help":
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set. If set to a float, will truncate the number of examples to that percentage of the dataset."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help":
            "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    dataset: str = field(
        default='alpaca', metadata={"help": "Which dataset to finetune on. See datamodule for options."}
    )
    dataset_format: Optional[str] = field(
        default="conversations",
        metadata={"help": "Which dataset format is used. [alpaca|conversations|chip2|self-instruct|hh-rlhf|mistral]"}
    )
    prompt_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "Which prompt type to use. [alpaca|chatlm|llama2|minicpm|conversations|chip2|self-instruct|hh-rlhf|mistral]"
        }
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    do_train: bool = field(default=True, metadata={"help": 'To train or not to train, that is the question?'})
    report_to: str = field(default="none")
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(
        default="paged_adamw_32bit"
    )  # "paged_lion_8bit", "paged_adamw_8bit", "paged_lion_32bit", "paged_adamw_32bit"
    lr_scheduler_type: str = field(
        default="constant_with_warmup"
    )  # "constant", "constant_with_warmup", "cosine", "cosine_with_restarts", "linear"
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )

    flash_attention: bool = field(default=True, metadata={"help": "Use flash attention."})

    bits: int = field(default=4, metadata={"help": "How many bits to use."})
    lora_r: int = field(default=64, metadata={"help": "Lora R dimension."})
    lora_alpha: float = field(default=16, metadata={"help": " Lora alpha."})
    lora_dropout: float = field(default=0.0, metadata={"help": "Lora dropout."})

    num_experts: int = field(default=8, metadata={"help": "Number of experts. 4/8/16"})
    topk: int = field(default=2, metadata={"help": "Top k. 2/2/4"})
    adapter_dim: int = field(default=64, metadata={"help": "Adapter dimension.64/64/512"})


def clean_memory():
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)
    torch.cuda.empty_cache()


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


class SavePeftModelCallback(transformers.TrainerCallback):

    def save_model(self, args, state, kwargs):
        logger.info('Saving PEFT checkpoint...')
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "adapter_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        model = kwargs["model"]
        model.save_pretrained(peft_model_path)

        moe_state = {}
        for param_tensor in model.state_dict():
            if "adapter" in param_tensor:
                moe_state.update({param_tensor: model.state_dict()[param_tensor]})
            # if "adapter" in param_tensor or "norm" in param_tensor:
            #     moe_state.update({param_tensor: model.state_dict()[param_tensor]})
        moe_model_path = os.path.join(checkpoint_folder, "moe_model.bin")
        # print(moe_state.keys())
        torch.save(moe_state, moe_model_path)

        self._symlink_latest_checkpoint(checkpoint_folder)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)

    def _symlink_latest_checkpoint(self, checkpoint_folder):
        # if the latest checkpoint is a symlink, remove it
        output_dir = os.path.dirname(checkpoint_folder)
        latest_checkpoint = os.path.join(output_dir, "latest")
        if os.path.islink(latest_checkpoint):
            os.remove(latest_checkpoint)
        # symlink the latest checkpoint to the checkpoint folder
        os.symlink(os.path.basename(checkpoint_folder), latest_checkpoint)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):

        def touch(fname, times=None):
            with open(fname, "a"):
                os.utime(fname, times)

        touch(join(args.output_dir, "completed"))
        self.save_model(args, state, kwargs)


def find_all_linear_names(model, bits=4):
    cls = (bnb.nn.Linear4bit if bits == 4 else (bnb.nn.Linear8bitLt if bits == 8 else torch.nn.Linear))
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.ddp_find_unused_parameters = False

    import argparse
    args = argparse.Namespace(**vars(model_args), **vars(data_args), **vars(training_args))
    set_seed(args.seed)

    print(f"{args=}")

    model_kwargs = {}
    if "mistral" in model_args.model_name_or_path:
        model_config = SparsetralConfig.from_pretrained(model_args.model_name_or_path)

        # Sparsetral Config
        model_config.moe_dtype = "bfloat16"
        model_config.lora_r = args.lora_r  #64
        model_config.lora_alpha = args.lora_alpha  #16
        model_config.adapter_dim = args.adapter_dim #512
        model_config.topk = args.topk # 4
        model_config.moe_scaling = 1
        model_config.num_experts = args.num_experts #16
        model_config.output_router_logits = True

        model_class = MistralForCausalLM

        if args.flash_attention:
            model_kwargs['attn_implementation'] = "flash_attention_2"

    # if "llama" in model_args.model_name_or_path:
    else:
        model_config = CamelidaeConfig.from_pretrained(model_args.model_name_or_path)
        model_config.pretraining_tp = 1  ## without tensor parallelism rank

        # Camelidae Config
        model_config.moe_dtype = "bfloat16"
        model_config.lora_r = args.lora_r  #64
        model_config.lora_alpha = args.lora_alpha  #16
        model_config.adapter_dim = args.adapter_dim # 64
        model_config.topk = args.topk # 2
        model_config.moe_scaling = 1
        model_config.num_experts = args.num_experts  # 8
        model_config.output_router_logits = False

        # # Seq Length Extension
        # model_config.rope_scaling = {
        #     "type": "dynamic",
        #     "factor": 2,
        # }

        model_class = LlamaForCausalLM

        if args.flash_attention:
            # from speechless.finetune.moe.camelidae.camelidae_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
            # print(f"Call camelidae Llama replace_llama_attn_with_flash_attn()")
            # replace_llama_attn_with_flash_attn()
            model_kwargs['attn_implementation'] = "flash_attention_2"


    model = model_class.from_pretrained(
        model_args.model_name_or_path,
        config=model_config,
        cache_dir=training_args.cache_dir,
        load_in_4bit=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ),
        output_loading_info=False,
        **model_kwargs,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.gradient_checkpointing_enable()

    # lora_modules = find_all_linear_names(model)
    lora_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "up_proj",
        "gate_proj",
        "down_proj",
    ]
    config = LoraConfig(
        r=model_config.lora_r,
        lora_alpha=model_config.lora_alpha,
        target_modules=lora_modules,
        lora_dropout=args.lora_dropout,  #0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    # Zero Init
    for n, p in model.named_parameters():
        if "adapter_up" in n:
            nn.init.zeros_(p)
        if "adapter_down" in n:
            nn.init.kaiming_uniform_(p, a=math.sqrt(5))
        if "router" in n:
            nn.init.kaiming_uniform_(p, a=math.sqrt(5))

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if training_args.bf16:
                module = module.to(torch.bfloat16)
        if "norm" in name:
            module = module.to(torch.float32)
        if "lm_head" in name or "embed_tokens" in name:
            if hasattr(module, "weight"):
                if training_args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
        if "adapter" in name:
            if training_args.bf16:
                module = module.to(torch.bfloat16)
            else:
                module = module.to(torch.float32)

    for n, p in model.named_parameters():
        if "adapter" in n:
            p.requires_grad = True
        # if "norm" in n:
        #     p.requires_grad = True

    model.config.use_cache = False
    print_trainable_parameters(model)

    # tokenizer = transformers.AutoTokenizer.from_pretrained(
    #     model_args.model_name_or_path,
    #     cache_dir=training_args.cache_dir,
    #     model_max_length=training_args.model_max_length,
    #     padding_side="left",
    #     use_fast=False,
    #     trust_remote_code=True,
    # )
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token_id = (
    #         0  # unk. we want this to be different from the eos token
    #     )

    # Tokenizer
    tokenizer_kwargs = {
        "cache_dir": args.cache_dir,
        "padding_side": "left",
        "use_fast": False,
        "trust_remote_code": True,
    }
    print(f"---------- Original tokens----------")
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path, **tokenizer_kwargs)
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
            tokenizer.pad_token_id = 0  # tokenizer.eos_token_id
            tokenizer.pad_token = tokenizer._convert_id_to_token(tokenizer.pad_token_id)  #tokenizer.eos_token
    print(f"---------- Fixed tokens ----------")
    print(f"{tokenizer.pad_token=},{tokenizer.pad_token_id=}")
    # print(f"{tokenizer.unk_token=},{tokenizer.unk_token_id=}")
    print(f"{tokenizer.bos_token=},{tokenizer.bos_token_id=}")
    print(f"{tokenizer.eos_token=},{tokenizer.eos_token_id=}")

    # data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    data_module = make_data_module(tokenizer=tokenizer, args=args)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **{k: v
           for k, v in data_module.items()
           if k != 'predict_dataset'},
    )
    trainer.add_callback(SavePeftModelCallback)

    trainer.add_callback(CleanMemoryCallback)
    trainer.add_callback(LoggingCallback)

    trainer.train()

    model.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train()
