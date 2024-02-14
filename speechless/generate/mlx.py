#!/usr/bin/env python
import os, json
import mlx_lm


def generate(args):
    model, tokenizer = mlx_lm.load(args.model_path, adapter_file=args.adapter_file)
    response = mlx_lm.generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        temp=args.temperature,
        max_tokens=args.max_tokens,
        verbose=args.verbose
    )
    print(response)


def get_args():
    from argparse import ArgumentParser

    parser = ArgumentParser(description="MLX: Machine Learning eXperiment")

    parser.add_argument("--model_path", type=str, required=True, help="HF model path")
    parser.add_argument("--adapter_file", type=str, help="adapter file path")
    parser.add_argument("--prompt", type=str, required=True, help="prompt to run")
    parser.add_argument("--temperature", type=float, default=0.7, help="temperature")
    parser.add_argument("--max_tokens", type=int, default=256, help="max tokens")
    parser.add_argument("--verbose", action="store_true", help="verbose")

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    generate(args)


if __name__ == '__main__':
    main()
