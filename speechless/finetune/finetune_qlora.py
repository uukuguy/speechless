# finetune_qlora.py
import os, math
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel
)
import bitsandbytes as bnb
from loguru import logger

from speechless.utils.model_utils import print_trainable_parameters, find_all_linear_names, get_accelerate_model
from speechless.finetune.finetune_qlora_arguments import get_args

def finetune():
    args = get_args()
    print(args)

if __name__ == "__main__":
    finetune()