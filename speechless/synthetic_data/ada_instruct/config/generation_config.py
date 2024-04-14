import sys
import os
import yaml
from pprint import pprint

from transformers import GenerationConfig


GENERATIONDICT_REGISTRY = {
    "humaneval": {
        "do_sample": True,
        "temperature": 1.0,
        "top_k": 80,
        "max_new_tokens": 256
    },
    "mbpp": {
        "do_sample": True,
        "temperature": 1.0,
        "top_k": 100,
        "max_new_tokens": 128
    },
    "gsm8k": {
        "do_sample": True,
        "temperature": 1.0,
        "top_k": 80,
        "max_new_tokens": 512
    },
    "math": {
        "do_sample": True,
        "temperature": 1.0,
        "top_k": 80,
        "max_new_tokens": 512
    },
    "csqa": {
        "do_sample": True,
        "temperature": 1.0,
        "top_k": 100,
        "max_new_tokens": 256
    }
}

def get_generation_config(generation_config_name, from_yml: str = ''):
    try:
        generation_dict = GENERATIONDICT_REGISTRY[generation_config_name]
        if from_yml:
            with open(from_yml, 'r', encoding='utf-8') as f:
                yml_dict = yaml.safe_load(f)
            generation_dict.update(yml_dict)
        return GenerationConfig(**generation_dict)
        
    except KeyError:
        print("Available generation configs:")
        pprint(GENERATIONDICT_REGISTRY)
        raise KeyError(f"Missing genertion config {generation_config_name}")