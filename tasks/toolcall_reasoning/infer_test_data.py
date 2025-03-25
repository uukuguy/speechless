#!/usr/bin/env python

import os, json, re
from tqdm import tqdm, trange
from loguru import logger
import rich

from speechless.api.llm_api import OpenAI_API
from speechless.utils.multiprocess_utils import initialize_multiprocessing, run_func_in_multiprocessing

def get_args():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)

    parser.add_argument("--parallel_processes", type=int, default=0, help="Number of parallel processes")
    parser.add_argument("--parallel_chunk_size", type=int, default=16, help="Chunk size for parallel processes")
    parser.add_argument("--request_batch_size", type=int, default=128, help="Number of requests in each batch")

    args = parser.parse_args()
    return args

def run_single(params):
    line = params["line"]
    llm_api = params["llm_api"]

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

    return data

def main():
    args = get_args()
    test_file = args.test_file

    llm_api = OpenAI_API(base_url=args.base_url, model_name=args.model_name)

    with open(test_file, "r") as f:
        lines = f.readlines()

    inferred_file = test_file.replace(".jsonl", "-inferred.jsonl")
    with open(inferred_file, "w", encoding="utf-8") as fd:
        # for line in tqdm(lines):
        #     data = json.loads(line)
        #     instruction = data["instruction"]
        #     apis = data["apis"]

        #     gen_kwargs = {
        #         "temperature": 0.95,
        #         "max_tokens": 8192,
        #         "frequency_penalty": 1.5,
        #         "stream": False,
        #         # "tool_choice": "auto",
        #     }
        #     response = llm_api(prompt_or_messages=instruction, gen_kwargs=gen_kwargs, tools=apis, verbose=False)
        #     if response is not None:
        #         generated_text = response.generated_text
        #         llm_response = response.llm_response
        #         data["generated_text"] = generated_text
        #         data["llm_response"] = llm_response if isinstance(llm_response, dict) else json.loads(llm_response.model_dump_json())
        #     else:
        #         data["generated_text"] = ""
        #         data["llm_response"] = {}

        #     f.write(json.dumps(data, ensure_ascii=False) + "\n")

        if args.parallel_processes > 1:
            request_batch_size = args.request_batch_size
            num_test_data = len(lines)
            for i in trange(0, len(lines), request_batch_size):
                batch_lines = lines[i:i+request_batch_size]
                params_list = [{"line": line, "llm_api": llm_api} for line in batch_lines]
                parallel_results = run_func_in_multiprocessing(
                    run_single,
                    params_list,
                    num_processes=args.parallel_processes,
                    chunk_size=args.parallel_chunk_size,
                    unordered=False,
                    use_progress_bar=True,
                    progress_bar_desc=f"Batch {i//request_batch_size}/{num_test_data//request_batch_size}",
                )
                for result in parallel_results:
                    rsp_messages, raw_messages = [], []
                    if result.is_success():
                        datas = result.result
                    fd.write(json.dumps(data, ensure_ascii=False) + "\n")
                    fd.flush()
        else:
            for line in tqdm(lines):
                data = json.loads(line)
                data = run_single({"line": line, "llm_api": llm_api})
                fd.write(json.dumps(data, ensure_ascii=False) + "\n")
                fd.flush()


if __name__ == "__main__":
    initialize_multiprocessing()
    main()