#!/usr/bin/env python
"""
Usage: python -m speechless.quant llamacpp --model_path /path/to/model --llamacpp_quant_type q4_k_m
"""
import os
import subprocess

def quant_by_llamacpp(args):
    model_path = args.model_path if args.model_path[-1] != "/" else args.model_path[:-1]
    llamacpp_root = args.llamacpp_root or (os.getenv("LLAMACPP_ROOT") or os.path.expanduser("~/llama.cpp"))
    gguf_dir = args.gguf_dir or model_path + "-GGUF"
    gguf_f16_file = args.gguf_f16_file or os.path.join(gguf_dir, f"{os.path.basename(model_path)}.f16.gguf")

    llamacpp_convert = f"{llamacpp_root}/convert.py"
    llamacpp_quantize = f"{llamacpp_root}/quantize"
    if not os.path.isdir(llamacpp_root) or not os.path.exists(llamacpp_convert) or not os.path.exists(llamacpp_quantize):
        raise Exception(f"llamacpp_root {llamacpp_root} not found or invalid")

    os.makedirs(gguf_dir, exist_ok=True)

    if not os.path.exists(gguf_f16_file):
        convert_cmd = f"python {llamacpp_convert} {model_path} --padvocab --outtype f16 --outfile {gguf_f16_file}"
        print(convert_cmd)
        os.system(convert_cmd)

    # ./quantize ${GGML_FILE} ${Q4_K_M_FILE} q4_k_m 
    quant_file = f"{gguf_dir}/{os.path.basename(model_path)}.{args.llamacpp_quant_type.upper()}.gguf"
    quant_cmd = f"{llamacpp_quantize} {gguf_f16_file} {quant_file} {args.llamacpp_quant_type}"
    print(quant_cmd)
    os.system(quant_cmd)

    print(f"{gguf_f16_file}")
    print(f"{quant_file}")

commands = {
    "llamacpp": quant_by_llamacpp,
}

llamacpp_quant_types = ["q4_k_m", "q5_k_m", "q8_0"]

def get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument("cmd", type=str, choices=commands.keys(), help="command to run")

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--gguf_dir", type=str)
    parser.add_argument("--gguf_f16_file", type=str)
    parser.add_argument("--llamacpp_quant_type", type=str, choices=llamacpp_quant_types)
    parser.add_argument("--llamacpp_root", type=str)

    parser.add_argument("--litellm_port", type=int, default=18341)

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    func = commands[args.cmd]
    func(args)


if __name__ == "__main__":
    main()
