#!/usr/bin/env python
import os, importlib
import ollama

def get_package_dir(pkg_name):
    spec = importlib.util.find_spec(pkg_name)
    if spec is not None:
        pkg_dir = os.path.dirname(spec.origin)
    else:
        pkg_dir = None
    return pkg_dir


def find_speechless_root():
    pkg_dir = os.path.dirname(get_package_dir("speechless")) or os.getenv("SPEECHLESS_ROOT") or os.curdir
    return pkg_dir


def do_lmeval_litellm(args):
    pass


def do_lmeval(args):
    speechless_root = find_speechless_root()
    speechless_eval_dir = f"{speechless_root}/speechless/eval"

    task_name = os.path.basename(args.model_path)

    if args.gen:
        eval_gen_cmd = f"python -m speechless.eval.lmeval --do_gen --model hf-causal-experimental --model_args pretrained=${args.model_path},use_accelerate=True --batch_size {args.batch_size} --write_out --output_path {args.output_dir}/lmeval/${task_name} "
        os.system(eval_gen_cmd)

    if args.show_result:
        cmd = f"python -m speechless.eval.lmeval --show_result --output_path {args.output_dir}/lmeval/${task_name}"
        os.system(eval_gen_cmd)


def do_humaneval(args):
    pass


def do_evalplus(args):
    pass


def do_multipl_e(args):
    pass


def do_bigcode(args):
    pass


commands = {
    "lmeval": do_lmeval,
    'humaneval': do_humaneval,
    "evalplus": do_evalplus,
    "multipl_e": do_multipl_e,
    "bigcode": do_bigcode,
}


def get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument("cmd", type=str, choices=commands.keys(), help="command to run")

    parser.add_argument("--gen", action="store_true")
    parser.add_argument("--show_result", action="store_true")

    parser.add_argument("--output_dir", type=str, default="eval_results")
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--batch_size", type=int, default=4)

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    func = commands[args.cmd]
    func(args)


if __name__ == "__main__":
    main()
