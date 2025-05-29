#!/usr/bin/env python

import os, json, re
from copy import deepcopy
from tqdm import tqdm, trange
from loguru import logger
from transformers import AutoTokenizer
import rich

from speechless.api.llm_api import OpenAI_API
from speechless.utils.multiprocess_utils import initialize_multiprocessing, run_func_in_multiprocessing
from speechless.utils import load_attribute_from_custom_module

# https://github.com/chujiezheng/chat_templates
available_chat_templates = [
    "auto",
    "alpaca",
    "chatml",
    "llama3",
    "qwen2.5"
]

def get_args():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default=None, help="Output file for generated answers.")
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--temperature", type=float, default=0.95, help="Temperature")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Max tokens")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top p")
    parser.add_argument("--min_p", type=float, default=None, help="Min p, should be 0.01 - 0.2")
    parser.add_argument("--presence_penalty", type=float, default=1.5, help="Presence penalty")
    parser.add_argument("--frequency_penalty", type=float, default=1.5, help="Frequency penalty")
    parser.add_argument("--stream", action="store_true", default=False, help="Stream")
    parser.add_argument("--tool_choice", type=str, default="auto", help="Tool choice")  

    parser.add_argument("--logits_processor_module_file", type=str, default=None, help="Logits processor module file")
    parser.add_argument("--logits_processor_class_name", type=str, default=None, help="Logits processor class name")

    parser.add_argument("--parallel_processes", type=int, default=4, help="Number of parallel processes")
    parser.add_argument("--parallel_chunk_size", type=int, default=16, help="Chunk size for parallel processes")
    parser.add_argument("--request_batch_size", type=int, default=64, help="Number of requests in each batch")

    parser.add_argument("--use_chat_template", type=str, default="auto", choices=available_chat_templates, help="Use chat template")
    parser.add_argument("--use_chat_template_file", type=str, default=None, help="Use chat template file")

    args = parser.parse_args()
    return args

args = get_args()
llm_api = OpenAI_API(base_url=args.base_url, model_name=args.model_name)
tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

def run_single(params):
    use_chat_template = params.get("use_chat_template")
    data = params["data"]
    gen_kwargs = deepcopy(params["gen_kwargs"])
    logits_processor_module_file = params.get("module_file")
    logits_processor_class_name = params.get("class_name")

    if logits_processor_module_file is not None and logits_processor_class_name is not None:
        logits_processor = load_attribute_from_custom_module(logits_processor_module_file, logits_processor_class_name)
        if logits_processor is not None:
            # gen_kwargs["logits_processors"] = [logits_processor(tokenizer)]
            gen_kwargs["extra_body"] = {
                "logits_processors": [logits_processor(tokenizer)],
            }
        else:
            raise ValueError(f"Logits processor {args.logits_processor_class_name} not found in {args.logits_processor_module_file}")

    id = data.get("id")
    instruction = data["instruction"]
    tools = data.get("tools")

    response = llm_api(prompt_or_messages=instruction, gen_kwargs=gen_kwargs, tools=tools, verbose=False)

    result = {}
    if response is not None:
        generated_text = response.generated_text
        llm_response = response.llm_response
        result["generated_text"] = generated_text
        result["llm_response"] = llm_response if isinstance(llm_response, dict) else json.loads(llm_response.model_dump_json())

        if args.verbose:
            # logger.info(f"Instruction: {instruction}")
            # logger.info(f"APIs: {apis}")
            logger.info(f"[{id=}] Generated text: {generated_text}")
            # logger.info(f"LLM response: {llm_response}")
    else:
        result = {
            "generated_text": "",
            "llm_response": {},
        }

    return result

def main():
    test_file = args.test_file

    gen_kwargs = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "presence_penalty": args.presence_penalty,
        "frequency_penalty": args.frequency_penalty,
        "stream": args.stream,
        "top_p": args.top_p,
        "min_p": args.min_p, 
        # "tool_choice": "auto",
    }

    # if args.logits_processor_module_file is not None and args.logits_processor_class_name is not None:
    #     logits_processor = load_attribute_from_custom_module(args.logits_processor_module_file, args.logits_processor_class_name)
    #     if logits_processor is not None:
    #         gen_kwargs["logits_processors"] = [logits_processor(tokenizer)]
    #     else:
    #         raise ValueError(f"Logits processor {args.logits_processor_class_name} not found in {args.logits_processor_module_file}")


    with open(test_file, "r") as f:
        lines = f.readlines()

    test_datas = [json.loads(line) for line in lines]

    if args.output_file is not None:
        inferred_file = args.output_file
    else:
        inferred_file = test_file.replace(".jsonl", "-inferred.jsonl")
    with open(inferred_file, "w", encoding="utf-8") as fd:
        if args.parallel_processes > 1:
            request_batch_size = args.request_batch_size
            num_test_data = len(lines)
            for i in trange(0, len(test_datas), request_batch_size):
                batch_datas = test_datas[i:i+request_batch_size]
                for idx, data in enumerate(batch_datas):
                        if "id" not in data:
                            data["id"] = str(i + idx)
                        if "messages" in data and args.use_chat_template == "auto":
                            data['instruction'] = tokenizer.apply_chat_template(data['messages'], add_generation_prompt=True, tokenize=False)
                params_list = [{"data": data, "gen_kwargs": gen_kwargs, "module_file": args.logits_processor_module_file, "class_name": args.logits_processor_class_name} for data in batch_datas]
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
            for i, data in enumerate(tqdm(test_datas)):
                if "id" not in data:
                    data["id"] = str(i)
                if "messages" in data and args.use_chat_template == "auto":
                    data['instruction'] = tokenizer.apply_chat_template(data['messages'], add_generation_prompt=True, tokenize=False)
                result = run_single({"data": data, "gen_kwargs": gen_kwargs})
                data["generated_text"] = result["generated_text"]
                data["llm_response"] = result["llm_response"]
                fd.write(json.dumps(data, ensure_ascii=False) + "\n")
                fd.flush()


if __name__ == "__main__":
    initialize_multiprocessing()
    main()