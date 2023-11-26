import argparse
import os, json
from tqdm import tqdm, trange
import torch
from transformers import LlamaTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig
from human_eval.data import write_jsonl, read_problems, stream_jsonl

from vllm import LLM
from vllm import SamplingParams

os.environ['RAY_memory_monitor_refresh_ms'] = '0'

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
    INSTRUCTION = f"""<INST>{input}</INST>"""
    return INSTRUCTION


def extract_code(completion):
    toks = completion.split("```")
    if len(toks) >= 3:
        code = "```" + toks[1] + "```"
    else:
        code = completion
    return code


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='bigcode/starcoder', help="")
    # parser.add_argument('--lora', type=str, default='bigcode/starcoder', help="")
    parser.add_argument('--output_file', type=str, help="")
    parser.add_argument('--start_index', type=int, default=0, help="")
    parser.add_argument('--end_index', type=int, default=164, help="")
    parser.add_argument('--temperature', type=float, default=0.8, help="")
    parser.add_argument('--N', type=int, default=1, help="")
    parser.add_argument('--num_seqs_per_iter', type=int, default=1, help='')
    parser.add_argument('--max_len', type=int, default=512, help="")
    parser.add_argument('--num_gpus', type=int, default=0, help="")
    parser.add_argument('--decoding_style', type=str, default='sampling', help="")
    parser.add_argument('--overwrite', action='store_true', help='')
    parser.add_argument('--sliding_window', type=int, default=0, help='')
    parser.add_argument('--gen_batch_size', type=int, default=4, help='')

    args = parser.parse_args()

    if args.num_gpus == 0:
        args.num_gpus = torch.cuda.device_count()

    problems = read_problems()
    task_ids = sorted(problems.keys())[args.start_index:args.end_index]

    prompts = [problems[task_id]['prompt'] for task_id in task_ids]
    num_samples = len(prompts)
    print("Number of samples: {}".format(num_samples))

    llm = LLM(model=args.model, tensor_parallel_size=args.num_gpus, trust_remote_code=True)
    sampling_params = SamplingParams(temperature=args.temperature, top_p=1, max_tokens=args.max_len)

    print(f"Loaded {args.model}.")

    output_dir = os.path.dirname(args.output_file)
    os.makedirs(output_dir, exist_ok=True)
    fd = open(args.output_file, 'w')
    num_writed = 0
    for i in trange(num_samples // args.gen_batch_size + 1, ncols=100):

        # if os.path.exists(output_file) and not args.overwrite:
        #     print(f'Skip {output_file} as it already exists')
        #     continue

        ids_batch = []
        prompt_batch = []
        for j in range(args.gen_batch_size):
            n = i * args.gen_batch_size + j
            if n >= num_samples:
                break
            prompt = prompts[n].replace('    ', '\t')
            prompt_batch.append(generate_alpaca_prompt(prompt))
            ids_batch.append(task_ids[n])

        if len(prompt_batch) == 0:
            break
        if args.decoding_style == 'sampling':
            loops = int(args.N / args.num_seqs_per_iter)
        else:
            loops = 1

        for _ in range(loops):

            with torch.no_grad():
                completions = llm.generate(prompt_batch, sampling_params)
            gen_seqs = [completion.outputs[0].text for completion in completions]

            if gen_seqs is not None:
                for seq_idx, (task_id, prompt, gen_seq) in enumerate(zip(ids_batch, prompt_batch, gen_seqs)):
                    completion_seq = gen_seq.split("### Response:")[-1]

                    completion_seq = completion_seq.replace('\t', '    ')
                    completion_seq = extract_code(completion_seq)

                    all_code = gen_seq.replace('\t', '    ')

                    result = {
                        'task_id': task_id,
                        'completion': completion_seq,
                        'all_code': all_code,
                    }
                    fd.write(f"{json.dumps(result, ensure_ascii=False)}\n")
                    num_writed += 1
    fd.close()
    print(f"Gnerated {num_writed} samples, and saved to {args.output_file}")


if __name__ == '__main__':
    main()
