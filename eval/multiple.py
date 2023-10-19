"""
From MultiPL-E/automodel.py

This script produces completions for roughly any AutoModelForCausalLM.
"""
import os, sys 
import copy
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, f"{script_path}/MultiPL-E")
from multipl_e.completions import make_main, stop_at_stop_token, partial_arg_parser

class Model:
    def __init__(self, name, revision, tokenizer_name=None, tokenizer_revision=None):
        dtype = torch.float16
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        # self.model = AutoModelForCausalLM.from_pretrained(name, revision=revision, torch_dtype=dtype, trust_remote_code=True).cuda()
        self.model = AutoModelForCausalLM.from_pretrained(name, revision=revision, torch_dtype=dtype, trust_remote_code=True, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or name, revision=tokenizer_revision or revision, padding_side="left", trust_remote_code=True)
        self.tokenizer.pad_token = "<|endoftext|>"
        
    def completion_tensors(
        self,
        prompts: list,
        max_length: int,
        temperature: float,
        top_p: float,
    ):
        inputs = self.tokenizer(prompts, padding=True, return_tensors="pt", return_token_type_ids=False).to("cuda")
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                do_sample=True,
                use_cache=True,
                top_p=top_p,
                temperature=temperature,
                max_length=max_length,
                pad_token_id=self.tokenizer.pad_token_id
            )
        return output

    def decode_single_output(self, output_tensor, prompt):
        # NOTE(arjun): skip_special_tokens=True is the convenient way to strip out the left-side
        # padding tokens.
        detok_hypo_str = self.tokenizer.decode(
            output_tensor, clean_up_tokenization_spaces=False, skip_special_tokens=True,
        )
        # Skip the prompt (which may even have stop_tokens)
        return detok_hypo_str[len(prompt) :]

    def completions(
        self, prompts: str, max_tokens: int, temperature: float, top_p, stop
    ):
        prompts = [ prompt.strip() for prompt in prompts ]
        output_tensors = self.completion_tensors(
            prompts,
            max_tokens,
            temperature,
            top_p,
        )
        return [
            stop_at_stop_token(self.decode_single_output(output_tensor, prompt), stop + ["<|endoftext|>"])
            for (prompt, output_tensor) in zip(prompts, output_tensors)
        ]

def automodel_partial_arg_parser():
    """
    This is also used by peftmodel.py.
    """
    args = partial_arg_parser()
    args.add_argument("--name", type=str, required="generate" in sys.argv)
    args.add_argument("--revision", type=str)
    args.add_argument("--tokenizer_name", type=str)
    args.add_argument("--tokenizer_revision", type=str)
    args.add_argument("--name-override", type=str)

    args.add_argument("--langs", type=str, nargs='+', default=None)
    args.add_argument("--results_dir", type=str)
    args.add_argument("cmd", type=str, choices=['generate', 'eval', 'results'], help="commands: ['generate', 'eval', 'results']")
    return args

def do_name_override(args):
    """
    Applies the --name-override flag, or uses the model name, correcting / and - which the rest of
    the toolchain does not like.
    """
    if args.name_override:
        name = args.name_override
    else:
        name = args.name.replace("/", "_").replace("-", "_")
    return name

def main():
    args = automodel_partial_arg_parser()
    args = args.parse_args()
    if args.cmd == "generate":
        model = Model(args.name, args.revision, args.tokenizer_name, args.tokenizer_revision)
        name = do_name_override(args)
        if args.langs is not None:
            for lang in tqdm(args.langs, ncols=100):
                task_args = copy.deepcopy(args)
                task_args.lang = lang
                make_main(task_args, name, model.completions)
        else:
            make_main(args, name, model.completions)

        os.makedirs(args.output_dir_prefix, exist_ok=True)
            
    elif args.cmd == "eval":
        # MULTIPL_E_RESULTS_DIR=$(basename ${PWD})
        # docker run -it --rm \
        #     --network none \
        #     -v ${PWD}:/${MULTIPL_E_RESULTS_DIR}:rw \
        #     multipl-e-eval --dir /${MULTIPL_E_RESULTS_DIR} --output-dir /${MULTIPL_E_RESULTS_DIR} --recursive
        real_path = os.path.realpath(args.results_dir)
        basename = os.path.basename(args.results_dir)
        basename = basename.replace("/", "_").replace("-", "_")
        docker_cmd = f"docker run -it --rm --network none -v {real_path}:/{basename}:rw multipl-e-eval --dir /{basename} --output-dir /{basename} --recursive"
        os.system(docker_cmd)
    elif args.cmd == "results":
        # python ${PWD}/eval/MultiPL-E/pass_k.py -k 1 ${MULTIPL_E_RESULTS_DIR}/*
        speechless_root = os.environ.get("SPEECHLESS_ROOT", os.curdir)
        os_cmd = f"python {speechless_root}/speechless/eval/MultiPL-E/pass_k.py -k 1 {args.results_dir}/*"
        os.system(os_cmd)

"""
MULTIPL_E_RESULTS_DIR=eval_results/multipl_e/${TASK_NAME}

MULTIPLE_E_LANG=mkdir -p ${MULTIPL_E_RESULTS_DIR} && \
	python eval/MultiPL-E/automodel.py \
		--name ${TEST_MODEL_PATH} \
		--root-dataset humaneval \
		--temperature 0.2 \
		--batch-size 20 \
		--completion-limit 20 \
		--output-dir-prefix ${MULTIPL_E_RESULTS_DIR} 
"""
if __name__ == "__main__":
    main()
