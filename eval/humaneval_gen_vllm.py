import argparse
import pprint
import sys
import os
import re
from tqdm import tqdm
import torch
from transformers import LlamaTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig
from human_eval.data import write_jsonl, read_problems, stream_jsonl

from vllm import LLM
from vllm import SamplingParams

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass
def generate_alpaca_prompt(input):
    INSTRUCTION = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.


### Instruction:
Create a Python script for this problem:
{input}

### Response:"""
    return INSTRUCTION

def generate_llama2_prompt(input):
    INSTRUCTION=f"""<INST>{input}</INST>"""
    return INSTRUCTION

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='bigcode/starcoder', help="")
    parser.add_argument('--lora', type=str, default='bigcode/starcoder', help="")
    parser.add_argument('--output_path', type=str, help="")
    parser.add_argument('--start_index', type=int, default=0, help="")
    parser.add_argument('--end_index', type=int, default=164, help="")
    parser.add_argument('--temperature', type=float, default=0.8, help="")
    parser.add_argument('--N', type=int, default=200, help="")
    parser.add_argument('--max_len', type=int, default=512, help="")
    parser.add_argument('--num_gpus', type=int, default=4, help="")
    parser.add_argument('--decoding_style', type=str, default='sampling', help="")
    parser.add_argument('--num_seqs_per_iter', type=int, default=50, help='')
    parser.add_argument('--overwrite', action='store_true', help='')

    args = parser.parse_args()

    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    problems = read_problems()

    task_ids = sorted(problems.keys())[args.start_index: args.end_index]
    prompts = [problems[task_id]['prompt'] for task_id in task_ids]
    num_samples = len(prompts)
    print("Number of samples: {}".format(num_samples))

    llm = LLM(model=args.model, tensor_parallel_size=args.num_gpus, trust_remote_code=True)
    sampling_params = SamplingParams(temperature=args.temperature, top_p=1, max_tokens=args.max_len)

    print(f"Loaded {args.model}.")
    output_file = args.output_path + '/human_eval_samples.jsonl'
    for i in tqdm(range(num_samples), ncols=0, total=num_samples):
        # output_file = args.output_path + '/{}.jsonl'.format(args.start_index + i)

        # if os.path.exists(output_file) and not args.overwrite:
        #     print(f'Skip {output_file} as it already exists')
        #     continue

        prompt = prompts[i].replace('    ', '\t')
        # prompt = "<INST>" + prompt + "</INST>"
        prompt_batch = [generate_alpaca_prompt(prompt)]
        print(f"{prompt_batch=}")

        ids_batch = [task_ids[i]]
        completion_seqs = []

        if args.decoding_style == 'sampling':
            loops = int(args.N / args.num_seqs_per_iter)
        else:
            loops = 1

        for _ in tqdm(range(loops), total=loops, leave=False, ncols=0):

            with torch.no_grad():
                completions = llm.generate(prompt_batch, sampling_params)
            gen_seqs = [completions[0].outputs[0].text]

            if gen_seqs is not None:
                assert len(ids_batch) == 1
                task_id = ids_batch[0]

                for seq_idx, gen_seq in enumerate(gen_seqs):
                    completion_seq = gen_seq.split("### Response:")[-1]
                    completion_seq = completion_seq.replace('\t', '    ')
                    all_code = gen_seq.replace('\t', '    ')

                    completion_seqs.append(
                        {'task_id': task_id,
                         'completion': completion_seq,
                         'all_code': all_code,
                         }
                    )

        # print("Saving results to {}".format(output_file))
        write_jsonl(output_file, completion_seqs, append=True)


if __name__ == '__main__':
    main()