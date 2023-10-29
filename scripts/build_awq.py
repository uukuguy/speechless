#!/usr/bin/env python

import os
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

def build_awq(args):
    model_path = args.model_path
    w_bit = 4
    group_size = 128
    quant_config = { "zero_point": True, "q_group_size": group_size, "w_bit": w_bit, "version": "GEMM" }

    quant_path = args.quant_path
    if quant_path is None:
        quant_path = os.path.dirname(model_path) + "/" + os.path.basename(model_path) + f"-w{w_bit}g{group_size}-awq"

    # Load model
    # NOTE: pass safetensors=True to load safetensors
    print(f"Loading model from {model_path}")
    model = AutoAWQForCausalLM.from_pretrained(model_path, **{"low_cpu_mem_usage": True})
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Quantize
    print(f"Quantizing model ......")
    model.quantize(tokenizer, quant_config=quant_config)

    # Save quantized model
    # NOTE: pass safetensors=True to save quantized model weights as safetensors
    print(f"Saving quantized model to {quant_path}")
    model.save_quantized(quant_path)
    tokenizer.save_pretrained(quant_path)

    print(f'Model is quantized and saved at "{quant_path}"')


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None, required=True)
    parser.add_argument('--quant_path', type=str, default=None)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    build_awq(args)