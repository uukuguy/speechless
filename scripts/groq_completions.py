#!/usr/bin/env python
import os, json
from dotenv import load_dotenv
import rich
from groq import Groq
from rich.markdown import Markdown
from rich.console import Console
# from Ipython.display import Markdown

load_dotenv(override=True, verbose=True)

console = Console()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
groq_models = [ "Llama3-8b-8192", "Llama3-70b-8192", "Mixtral-8x7b-32768"]

def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Llama3-70b-8192", choices=groq_models)
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--prompt_file", type=str)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def do_chat(args):
    model = args.model
    prompt = args.prompt
    if args.prompt_file:
        if os.path.exists(args.prompt_file):
            prompt = open(args.prompt_file).read()
    print(f"{prompt=}")
    if prompt:
        result = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt},
            ])
        if args.verbose:
            print(f"{result=}")
        response_content = result.choices[0].message.content
        console.print(Markdown(response_content))
    else:
        raise ValueError("Prompt or prompt file must be provided.")

if __name__ == "__main__":
    args = get_args()
    do_chat(args)