#!/usr/bin/env python
import torch
import os, sys, json
from tqdm import tqdm
from copy import deepcopy
from loguru import logger

import multiprocessing as mp
mp.set_start_method('spawn', force=True)

SPEECHLESS_ROOT = os.getenv("SPEECHLESS_ROOT")
if SPEECHLESS_ROOT:
    sys.path.append(SPEECHLESS_ROOT)
else:
    sys.path.append("../../speechless")

def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="Path to the model.")
    parser.add_argument("--output_file", type=str, default="./inference_results.jsonl", help="Output file for generated answers.")
    parser.add_argument("--test_file", type=str, help="Input test file.")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Maximum number of tokens to generate.")
    parser.add_argument("--tensor_parallel_size", type=int, default=0, help="Tensor parallel size.")
    parser.add_argument("--batch_size", type=int, default=0, help="Batch size for generation.")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature for sampling.")
    return parser.parse_args()

def main():
    args = get_args()
    model_path = args.model_path
    test_file = args.test_file
    output_file = args.output_file
    temperature = args.temperature
    max_tokens = args.temperature

    tensor_parallel_size = args.tensor_parallel_size
    if tensor_parallel_size == 0:
        tensor_parallel_size = torch.cuda.device_count()
    batch_size = args.batch_size
    if batch_size == 0:
        batch_size = tensor_parallel_size * 4 

    test_data = [json.loads(line.strip()) for line in open(test_file).readlines()]

    from speechless.generate.llm import VllmAIModel, HFAIModel
    # vllm
    model = VllmAIModel(model_path=model_path, max_tokens=max_tokens, tensor_parallel_size=tensor_parallel_size)
    sampling_params = {
        'temperature': temperature,
        'max_tokens': max_tokens,
        # 'use_beam_search': False,
        # 'repetition_penalty': 1.0,
        # 'length_penalty': 1.0,
        # 'top_p': 1.0,
        # 'top_k':  -1,
        # 'seed': None,
        # 'stop': = None,
        # 'skip_special_tokens': True,
    }

    # # hf
    # gen_kwargs = {
    #     "temperature": temperature, 
    #     "max_new_tokens": max_tokens,
    #     "do_sample": True,
    #     # "top_p": 0.9,
    #     "min_p": 0.1,
    # }
    # model = HFAIModel(model_path=model_path, max_tokens=max_tokens, bits=8, gen_kwargs=gen_kwargs)


    prompts = [data['instruction'] for data in test_data]
    with open(output_file, 'w', encoding='utf-8') as fd:
        for s, e, batch_responses in model.generate_batch(prompts, batch_size=batch_size, **sampling_params):
        # for s, e, batch_responses in model.generate_batch(prompts, batch_size=max_examples):
            assert e - s == len(batch_responses), f"{s=}, {e=}, {len(batch_responses)=}"
            # for response, instruction, raw_data in zip(batch_responses, prompts[s:e], raw_datas[s:e]):
            for response, instruction, raw_data in zip(batch_responses, prompts[s:e], test_data[s:e]):
                logger.debug(f"{instruction=}")
                logger.info(f"{response=}")
                raw_data = deepcopy(raw_data)
                raw_data['response'] = response
                fd.write(json.dumps(raw_data, ensure_ascii=False) + "\n")
                fd.flush()

    print(f"Saved {len(test_data)} lines in {output_file}")

if __name__ == "__main__":
    main()