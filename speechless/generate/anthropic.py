#!/usr/bin/env python
"""
Usage:  python -m speechless.generate.anthropic -model claude-3-haiku-20240307 --verbose --max_tokens 512 --prompt_file $SPEECHLESS_ROOT/speechless/generate/prompts/hello_llm_en.txt
"""

import os
from anthropic import Anthropic

def generate(args):
    if args.prompt_file:
        prompt = open(args.prompt_file).read().strip()
    else:
        prompt = args.prompt

    client = Anthropic(base_url=os.getenv("ANTHROPIC_API_BASE"), api_key=os.getenv("ANTHROPIC_API_KEY"))

    generate_kwargs = {
        "temperature": args.temperature,
        "max_tokens_to_sample": args.max_tokens,
        "top_p": args.top_p,
    }
    if args.json_mode:
        generate_kwargs['response_format'] = "json_object"

    system_message = args.system_message or "You are a helpful assistant."
    response = client.completions.create(
        model=args.model,
        prompt=prompt,
        # messages=[
        #     {"role": "system", "content": system_message },
        #     {"role": "user", "content": prompt},
        # ],
        **generate_kwargs,
    )

    print(f"{response=}")
    print(response.content)

def get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    # gpt-3.5-turbo-1106, gpt-4-0613, claude-3-haiku-20240307, claude-3-opus-20240229
    parser.add_argument("--model", type=str, default="claude-3-haiku-20240307", help="Model name")
    parser.add_argument("--prompt", type=str, help="prompt to run")
    parser.add_argument("--prompt_file", type=str, help="prompt file")
    parser.add_argument("--system_message", type=str, default="You are a helpful assistant.", help="system message")

    parser.add_argument("--temperature", type=float, default=0.7, help="temperature")
    parser.add_argument("--max_tokens", type=int, default=2048, help="max tokens")

    parser.add_argument("--top_p", type=float, default=0.9, help="top p")
    parser.add_argument("--json_mode", action="store_true", help="json mode")


    parser.add_argument("--verbose", action="store_true", help="verbose")
    parser.add_argument("--stop", type=str, help="stop token")
    parser.add_argument("--seed", type=int, help="seed")

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    generate(args)


if __name__ == '__main__':
    main()
