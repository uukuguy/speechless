#!/usr/bin/env python
"""
Usage:  python -m speechless.generate.openai --model gpt-3.5-turbo-1106 --verbose --max_tokens 512 --prompt_file $SPEECHLESS_ROOT/speechless/generate/prompts/hello_llm_en.txt
"""

import os
import openai
# https://api.openai.com/v1/chat/completions

# openai.base_url=os.getenv("OPENAI_API_BASE")
# openai.api_key=os.getenv("OPENAI_API_KEY")
# print(f"{openai.api_base=}")

def generate(args):
    if args.prompt_file:
        prompt = open(args.prompt_file).read().strip()
    else:
        prompt = args.prompt

    client = openai.OpenAI()#base_url=os.getenv("OPENAI_API_BASE"))

    generate_kwargs = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "top_p": args.top_p,
        "stop": args.stop,
        "seed": args.seed,
        "stream": args.stream,
    }
    if args.json_mode:
        generate_kwargs['response_format'] = "json_object"

    system_message = args.system_message or "You are a helpful assistant."
    response = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": system_message },
            {"role": "user", "content": prompt},
        ],
        **generate_kwargs,
    )

    if args.stream:
        for chunk in response:
            print(chunk.choices[0].delta.content, end="")
    else:
        # print(f"{response=}")
        print(response.choices[0].message.content)

def get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    # gpt-3.5-turbo-1106, gpt-4-0613
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo-1106", help="Model name")
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
    parser.add_argument("--stream", action="store_true", help="streaming mode")

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    generate(args)


if __name__ == '__main__':
    main()
