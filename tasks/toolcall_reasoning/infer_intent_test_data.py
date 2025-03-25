#!/usr/bin/env python

import os, json, re
from tqdm import tqdm
from loguru import logger
import rich

test_file = "./toolcall_data/6_3_1/toolcall-instructions-intent-test-1000-1000-6_3_1.jsonl"

from speechless.api.llm_api import OpenAI_API

def get_args():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)

    args = parser.parse_args()
    return args

def main():
    args = get_args()

    llm_api = OpenAI_API(base_url=args.base_url, model_name=args.model_name)

    with open(test_file, "r") as f:
        lines = f.readlines()

    inferred_file = test_file.replace(".jsonl", "-inferred.jsonl")
    with open(inferred_file, "w", encoding="utf-8") as f:
        for line in tqdm(lines):
            data = json.loads(line)
            instruction = data["instruction"]
            apis = data["apis"]

            gen_kwargs = {
                "temperature": 0.95,
                "max_tokens": 8192,
                "frequency_penalty": 1.5,
                "stream": False,
                # "tool_choice": "auto",
            }
            response = llm_api(prompt_or_messages=instruction, gen_kwargs=gen_kwargs, tools=apis, verbose=False)
            if response is not None:
                generated_text = response.generated_text
                llm_response = response.llm_response
                data["generated_text"] = generated_text
                data["llm_response"] = llm_response if isinstance(llm_response, dict) else json.loads(llm_response.model_dump_json())
            else:
                data["generated_text"] = ""
                data["llm_response"] = {}

            f.write(json.dumps(data, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()