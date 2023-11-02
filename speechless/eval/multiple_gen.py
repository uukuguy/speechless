#!/usr/bin/env python

import os, json, time, copy
from tqdm import tqdm
from loguru import logger
from datasets import load_dataset

from async_openai import batch_completion, CompletionPrompt, default_argument_parser, sampling_params_from_args

def get_args():
    parser = default_argument_parser()

    available_langs = ['py', 'java', 'js', 'cpp', 'rs', 'go', 'sh', 'jl', 'ts']

    parser.add_argument("--do_generate", action="store_true")
    parser.add_argument("--do_convert", action="store_true")
    parser.add_argument("--output_dir", type=str, default="./output_multiple_gen")
    parser.add_argument("--root_dataset", type=str, default='humaneval', choices=['humaneval', 'mbpp']) 
    parser.add_argument("--langs", type=str, nargs='+', default=available_langs)
    parser.add_argument("-f", "--prompt_file", type=str, help="The path to the file containing the prompt template.")

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"eval_results/multipl_e/{args.model_name}"

    return args

def generate_speechless_prompt(input):
#     INSTRUCTION = f"""<s>A Bot

# USER: {input}

# ASSISTANT: """
    INSTRUCTION = f"""<s>{input}"""
    return INSTRUCTION

def build_completion_prompts(dataset, prompt_template, sampling_params):
    completion_prompts = []
    for data in dataset:
        s_params = copy.deepcopy(sampling_params)
        s_params['stop'] = data['stop_tokens']
        prompt = data['prompt']
        # prompt = generate_speechless_prompt(prompt)
        completion_prompt = CompletionPrompt(
            completion_type="commpletion",
            prompt_template=prompt,
            prompt_args={},
            input_data=data,
            sampling_params=s_params,
        ) 
        completion_prompts.append(completion_prompt)
    

    return completion_prompts

def do_convert(args):
    # output_file = f"{args.output_dir}/multiple-generated-{data_dir}.jsonl"
    import gzip
    from glob import glob
    files = glob(f"{args.output_dir}/multiple-generated-*.jsonl")
    for file in tqdm(files, ncols=100):
        s_dir = file.replace(".jsonl", "")
        s_dir = os.path.dirname(s_dir) + "/multiple/" + os.path.basename(s_dir)
        os.makedirs(s_dir, exist_ok=True)

        samples = [json.loads(line.strip()) for line in open(file).readlines()]
        for s in tqdm(samples, ncols=100, desc=f"{os.path.basename(file)}"):
            sampling_params = s['sampling_params']
            input_data = s['input_data']
            problem_name = input_data['name']

            completions = []
            for result_dict in s['result_dicts']:
                completions.append(result_dict['completion'])

            multiple_data = {
                'name': problem_name,
                'language': input_data['language'],
                'temperature': sampling_params['temperature'],
                'top_p': sampling_params['top_p'],
                'max_tokens': sampling_params['max_tokens'],
                'prompt': input_data['prompt'],
                'tests': input_data['tests'],
                'completions': completions,
                'stop_tokens': input_data['stop_tokens'],
            }
            s_file = f"{s_dir}/{problem_name}.json.gz"
            with gzip.open (s_file, 'wb') as fd:
                fd.write(bytes(json.dumps(multiple_data, ensure_ascii=False), 'utf-8'))

def do_generate(args):
    start_time = time.time()
    logger.info(f"Starting all languages {args.langs}...")

    prompt_template = None
    if args.prompt_file: 
        with open(args.prompt_file) as file:
            prompt_template = file.read()

    os.makedirs(args.output_dir, exist_ok=True)

    pbar = tqdm(args.langs, ncols=100)
    for lang in pbar:
        lang_time = time.time()

        data_dir = f"{args.root_dataset}-{lang}"
        pbar.set_description(data_dir)

        n_retries = 0
        while n_retries < 3:
            try:
                lang_dataset = load_dataset("nuprl/MultiPL-E", data_dir, split="test")
                break
            except Exception as e:
                logger.warning(f"Failed to load dataset {data_dir}: {e}")
                n_retries += 1

        sampling_params = sampling_params_from_args(args)
        input_completion_prompts = build_completion_prompts(lang_dataset, prompt_template=prompt_template, sampling_params=sampling_params)

        output_completion_prompts = batch_completion(
            model_name=args.model_name,
            input_completion_prompts=input_completion_prompts,
            sampling_params=sampling_params,
            parallel_threads=args.parallel_threads,
            timeout=args.timeout,
            completion_limit=args.completion_limit,
            )

        lang_end_time = time.time()
        logger.info(f"Finished {lang} in {lang_end_time - lang_time:.2f}s")

        output_file = f"{args.output_dir}/multiple-generated-{data_dir}.jsonl"
        logger.info(f"Saving {output_file} ...")
        with open(output_file, "a") as fd:
            for completion_prompt in output_completion_prompts:
                fd.write(f"{json.dumps(completion_prompt.__dict__, ensure_ascii=False)}\n")
            logger.info(f"Saved {len(output_completion_prompts)} completion prompts to {output_file}")

    logger.info(f"Finished all {len(args.langs)} languages in {time.time() - start_time:.2f}s")

def main():
    args = get_args()
    print(f"{args=}")

    if args.do_generate:
        do_generate(args)
    if args.do_convert:
        do_convert(args)


if __name__ == '__main__':
    main()
