#!/usr/bin/env python
"""
Usage:  python -m speechless.generate.llamacpp --model_path sus-chat-34b.Q8_0.gguf --prompt 写一篇春节拜年的小学作文
"""

import os


def generate(args):
    BASE_OPTS = "--keep 0 --mirostat 2 --color --repeat-last-n 256"
    DISABLE_LOG = "--log-disable"
    GENERATE_OPTS = f"--temp {args.temperature} --top-k {args.top_k} --top-p {args.top_p} --repeat_penalty {args.repeat_penalty}"
    LLAMA_CPP_ROOT = os.getenv("LLAMACPP_ROOT") or os.path.expanduser("~/llama.cpp")

    if args.prompt_file:
        prompt = open(args.prompt_file).read().strip()
    else:
        prompt = args.prompt

    cmd = f"{LLAMA_CPP_ROOT}/main -m '{args.model_path}' -ngl {args.ngl} -c {args.ctx_size} -n {args.max_tokens} {'' if args.verbose else DISABLE_LOG} {BASE_OPTS} {GENERATE_OPTS} -p '{prompt}' "

    # print("Running command: ", cmd)

    os.system(cmd)


def get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument("--model_path", type=str, required=True, help="GGUF file")
    parser.add_argument("--prompt", type=str, help="prompt to run")
    parser.add_argument("--prompt_file", type=str, help="prompt file")
    parser.add_argument("--max_tokens", type=int, default=16384, help="max tokens")
    parser.add_argument("--ctx_size", type=int, default=16384, help="context size")
    parser.add_argument("--temperature", type=float, default=0.7, help="temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="top p")
    parser.add_argument("--top_k", type=int, default=40, help="top k")
    parser.add_argument("--repeat_penalty", type=float, default=1.1, help="repeat penalty")
    parser.add_argument("--verbose", action="store_true", help="verbose")
    parser.add_argument("--ngl", type=int, default=512, help="number of layers on GPU")

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    generate(args)


if __name__ == '__main__':
    main()
