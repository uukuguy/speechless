import os
from typing import TYPE_CHECKING, Any, Dict, Tuple
import torch

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .adapter import init_adapter
from .patcher import patch_config, patch_model, patch_tokenizer #, patch_valuehead_model
from .utils import load_valuehead_params, register_autoclass


# from ..extras.misc import count_parameters, get_current_device, try_download_model_from_ms
from .utils import get_current_device
def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    r"""
    Returns the number of trainable parameters and number of all parameters in the model.
    """
    trainable_params, all_param = 0, 0
    for param in model.parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes, multiply the number of parameters by 2
        if param.__class__.__name__ == "Params4bit":
            if hasattr(param, "quant_storage") and hasattr(param.quant_storage, "itemsize"):
                num_bytes = param.quant_storage.itemsize
            else:
                num_bytes = 1

            num_params = num_params * 2 * num_bytes

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param


if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer

    from ..hparams import FinetuningArguments, ModelArguments


# from ..extras.logging import get_logger
# logger = get_logger(__name__)
from loguru import logger

def use_modelscope() -> bool:
    return bool(int(os.environ.get("USE_MODELSCOPE_HUB", "0")))

def try_download_model_from_ms(model_args: "ModelArguments") -> str:
    if not use_modelscope() or os.path.exists(model_args.model_name_or_path):
        return model_args.model_name_or_path

    try:
        from modelscope import snapshot_download

        revision = "master" if model_args.model_revision == "main" else model_args.model_revision
        return snapshot_download(model_args.model_name_or_path, revision=revision, cache_dir=model_args.cache_dir)
    except ImportError:
        raise ImportError("Please install modelscope via `pip install modelscope -U`")


def _get_init_kwargs(model_args: "ModelArguments") -> Dict[str, Any]:
    model_args.model_name_or_path = try_download_model_from_ms(model_args)
    return {
        "trust_remote_code": True,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.hf_hub_token,
    }


def load_tokenizer(model_args: "ModelArguments") -> "PreTrainedTokenizer":
    r"""
    Loads pretrained tokenizer. Must before load_model.

    Note: including inplace operation of model_args.
    """
    init_kwargs = _get_init_kwargs(model_args)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
        split_special_tokens=model_args.split_special_tokens,
        padding_side="right",
        **init_kwargs,
    )
    patch_tokenizer(tokenizer)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0 # tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer._convert_id_to_token(tokenizer.pad_token_id) #tokenizer.eos_token

    return tokenizer


def load_model(
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool = False,
    add_valuehead: bool = False,
) -> "PreTrainedModel":
    r"""
    Loads pretrained model. Must after load_tokenizer.
    """
    init_kwargs = _get_init_kwargs(model_args)
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, **init_kwargs)
    patch_config(config, tokenizer, model_args, init_kwargs, is_trainable)

    model = None
    if is_trainable and model_args.use_unsloth:
        from unsloth import FastLanguageModel  # type: ignore

        unsloth_kwargs = {
            "model_name": model_args.model_name_or_path,
            "max_seq_length": model_args.model_max_length,
            "dtype": model_args.compute_dtype,
            "load_in_4bit": model_args.quantization_bit == 4,
            "token": model_args.hf_hub_token,
            "device_map": {"": get_current_device()},
            "rope_scaling": getattr(config, "rope_scaling", None),
        }
        try:
            model, _ = FastLanguageModel.from_pretrained(**unsloth_kwargs)
        except NotImplementedError:
            logger.warning("Unsloth does not support model type {}.".format(getattr(config, "model_type", None)))
            model_args.use_unsloth = False

        if model_args.adapter_name_or_path:
            model_args.adapter_name_or_path = None
            logger.warning("Unsloth does not support loading adapters.")

    if model is None:
        model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, config=config, **init_kwargs)

    patch_model(model, tokenizer, model_args, is_trainable)
    register_autoclass(config, model, tokenizer)

    model = init_adapter(model, model_args, finetuning_args, is_trainable)

    if add_valuehead:
        from trl import AutoModelForCausalLMWithValueHead
        model: "AutoModelForCausalLMWithValueHead" = AutoModelForCausalLMWithValueHead.from_pretrained(model)
        patch_valuehead_model(model)

        if model_args.adapter_name_or_path is not None:
            vhead_path = model_args.adapter_name_or_path[-1]
        else:
            vhead_path = model_args.model_name_or_path

        vhead_params = load_valuehead_params(vhead_path, model_args)
        if vhead_params is not None:
            model.load_state_dict(vhead_params, strict=False)
            logger.info("Loaded valuehead from checkpoint: {}".format(vhead_path))

    if not is_trainable:
        model.requires_grad_(False)
        model.eval()
        for param in model.parameters():
            if param.device.type == "cuda":
                param.data = param.data.to(model_args.compute_dtype)
    else:
        model.train()

    trainable_params, all_param = count_parameters(model)
    if is_trainable:
        param_stats = "trainable params: {:d} || all params: {:d} || trainable%: {:.4f}".format(
            trainable_params, all_param, 100 * trainable_params / all_param
        )
    else:
        param_stats = "all params: {:d}".format(all_param)
    logger.info(param_stats)

    if model_args.print_param_status:
        for name, param in model.named_parameters():
            print(
                "name: {}, dtype: {}, device: {}, trainable: {}".format(
                    name, param.dtype, param.device, param.requires_grad
                )
            )

    return model


def load_model_and_tokenizer(
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool = False,
    add_valuehead: bool = False,
) -> Tuple["PreTrainedModel", "PreTrainedTokenizer"]:
    r"""
    Loads pretrained model and tokenizer.
    """
    tokenizer = load_tokenizer(model_args)
    model = load_model(tokenizer, model_args, finetuning_args, is_trainable, add_valuehead)
    return model, tokenizer
