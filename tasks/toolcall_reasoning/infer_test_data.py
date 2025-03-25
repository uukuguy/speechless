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

    parser.add_argument("--parallel_processes", type=int, default=4, help="Number of parallel processes")
    parser.add_argument("--parallel_chunk_size", type=int, default=16, help="Chunk size for parallel processes")
    parser.add_argument("--request_batch_size", type=int, default=64, help="Number of requests in each batch")

    args = parser.parse_args()
    return args

args = get_args()
llm_api = OpenAI_API(base_url=args.base_url, model_name=args.model_name)

def run_single(params):
    data = params["data"]

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
    result = {
        "generated_text": "",
        "llm_response": {},
    }
    if response is not None:
        generated_text = response.generated_text
        llm_response = response.llm_response
        result["generated_text"] = generated_text
        result["llm_response"] = llm_response if isinstance(llm_response, dict) else json.loads(llm_response.model_dump_json())

    return result

def main():
    test_file = args.test_file

    with open(test_file, "r") as f:
        lines = f.readlines()

    test_datas = [json.loads(line) for line in lines]

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
            for i in trange(0, len(test_datas), request_batch_size):
                batch_datas = test_datas[i:i+request_batch_size]
                params_list = [{"data": data} for data in batch_datas]
                parallel_results = run_func_in_multiprocessing(
                    run_single,
                    params_list,
                    num_processes=args.parallel_processes,
                    chunk_size=args.parallel_chunk_size,
                    unordered=False,
                    use_progress_bar=True,
                    progress_bar_desc=f"Batch {i//request_batch_size}/{num_test_data//request_batch_size}",
                )
                for j, result in enumerate(parallel_results):
                    data = batch_datas[j]
                    if result.is_success():
                        result = result.result
                        data["generated_text"] = result["generated_text"]
                        data["llm_response"] = result["llm_response"]
                    else:
                        data["generated_text"] = ""
                        data["llm_response"] = {}
                    json_str = json.dumps(data, ensure_ascii=False) + "\n" 
                    fd.write(json_str)
                    fd.flush()
        else:
            for data in tqdm(test_datas):
                result = run_single({"data": data})
                data["generated_text"] = result["generated_text"]
                data["llm_response"] = result["llm_response"]
                fd.write(json.dumps(data, ensure_ascii=False) + "\n")
                fd.flush()


if __name__ == "__main__":
    initialize_multiprocessing()
    main()