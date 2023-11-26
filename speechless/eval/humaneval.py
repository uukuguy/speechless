import os, json
from tqdm import trange
import torch

from .human_eval.data import HUMAN_EVAL, read_problems
from .human_eval.evaluation import evaluate_functional_correctness

from vllm import LLM
from vllm import SamplingParams

os.environ['RAY_memory_monitor_refresh_ms'] = '0'
# huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
os.environ["TOKENIZERS_PARALLELISM"] = "true" 

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


def do_gen(args):

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

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    humaneval_samples_file = f"{output_dir}/humaneval_samples.jsonl"
    fd = open(humaneval_samples_file, 'w')
    num_writed = 0
    for i in trange(num_samples // args.gen_batch_size + 1, ncols=100):

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
        # if args.decoding_style == 'sampling':
        #     loops = int(args.N / args.num_seqs_per_iter)
        # else:
        #     loops = 1
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
    print(f"Gnerated {num_writed} samples, and saved to {humaneval_samples_file}")



def do_eval(args):
    sample_file = f"{args.output_dir}/humaneval_samples.jsonl"
    k = args.k
    n_workers = args.n_workers
    timeout = args.timeout
    problem_file = HUMAN_EVAL
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """
    k = list(map(int, k.split(",")))
    results = evaluate_functional_correctness(sample_file, k, n_workers, timeout, problem_file)
    print(results)

def get_args():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", type=str, default="eval_results/humaneval")

    parser.add_argument("--do_eval", action="store_true", help="")
    parser.add_argument("--k", type=str, default="1,10,100", help="")
    parser.add_argument("--n_workers", type=int, default=4, help="")
    parser.add_argument("--timeout", type=float, default=3.0, help="")

    parser.add_argument("--do_gen", action="store_true", help="")
    parser.add_argument('--model', type=str, default='bigcode/starcoder', help="")
    # parser.add_argument('--lora', type=str, default='bigcode/starcoder', help="")
    parser.add_argument('--start_index', type=int, default=0, help="")
    parser.add_argument('--end_index', type=int, default=164, help="")
    parser.add_argument('--temperature', type=float, default=0.2, help="")
    parser.add_argument('--max_len', type=int, default=512, help="")
    parser.add_argument('--num_gpus', type=int, default=0, help="")
    parser.add_argument('--decoding_style', type=str, default=None, help="")
    parser.add_argument('--gen_batch_size', type=int, default=16, help='')

    args = parser.parse_args()
    return args

def main():
    args = get_args()
    if args.do_gen:
        do_gen(args)
    if args.do_eval:
        do_eval(args)

if __name__ == "__main__":
    main()