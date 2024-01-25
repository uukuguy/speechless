"""
Usage:  python -m speechless.finetune init --task_name my_task
        python -m speechless.finetune run --task_name my_task
        python -m speechless.finetune merge --task_name my_task
        python -m speechless.finetune backup --task_name my_task
        python -m speechless.finetune list
"""
import os, importlib
import shutil


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

def is_speechless_task_dir(task_dir, additional_files=[]):
    speechless_run_finetune = f"{task_dir}/run_finetune.sh"
    speechless_taask_env = f"{task_dir}/task.env"
    if not os.path.exists(task_dir) or not os.path.exists(speechless_run_finetune) or not os.path.exists(speechless_taask_env):
        return False
    for file in additional_files:
        if not os.path.exists(f"{task_dir}/{file}"):
            print(f"{task_dir}/{file} not found")
            return False
    return True

def finetune_init(args):
    speechless_root = args.speechless_root
    default_task_dir = f"{speechless_root}/speechless/finetune/default_task"

    task_name = args.task_name
    if not task_name:
        raise Exception("task_name is required")
    if os.path.exists(task_name):
        print(f"{task_name} already exists")
        return
    task_dir = f"{os.curdir}/{task_name}"

    if not is_speechless_task_dir(default_task_dir):
        raise Exception(f"{default_task_dir} not found or invalid")

    shutil.copytree(default_task_dir, task_dir, dirs_exist_ok=False)

    print(f"Task {task_name} initialized in {task_dir}")


def finetune_run(args):
    task_name = args.task_name
    if not task_name:
        raise Exception("task_name is required")
    task_dir = f"{os.curdir}/{task_name}"

    if not is_speechless_task_dir(task_dir):
        raise Exception(f"{task_dir} not found or invalid")

    run_cmd = f"cd {task_dir} && ./run_finetune.sh"
    os.system(run_cmd)


def finetune_merge(args):
    task_name = args.task_name
    if not task_name:
        raise Exception("task_name is required")
    task_dir = f"{os.curdir}/{task_name}"

    if not is_speechless_task_dir(task_dir, ["merge.sh"]):
        raise Exception(f"{task_dir} not found or invalid")

    run_cmd = f"cd {task_dir} && ./merge.sh"
    os.system(run_cmd)


def finetune_backup(args):
    task_name = args.task_name
    if not task_name:
        raise Exception("task_name is required")
    task_dir = f"{os.curdir}/{task_name}"

    if not is_speechless_task_dir(task_dir, ["backup.sh"]):
        raise Exception(f"{task_dir} not found or invalid")

    run_cmd = f"cd {task_dir} && ./backup.sh"
    os.system(run_cmd)


def finetune_list(args):
    subdirs = [x for x in os.listdir(os.curdir) if os.path.isdir(x)]
    speechless_task_dirs = [x for x in subdirs if is_speechless_task_dir(x) ]
    for speechless_task_dir in speechless_task_dirs:
        print(speechless_task_dir)


commands = {
    "init": finetune_init,
    "run": finetune_run,
    "merge": finetune_merge,
    "backup": finetune_backup,
    "list": finetune_list,
}


def get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument("cmd", type=str, choices=commands.keys(), help="command to run")

    parser.add_argument("--task_name", type=str)
    parser.add_argument("--speechless_root", type=str, default=find_speechless_root(), help="Speechless root directory")

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    func = commands[args.cmd]
    func(args)


if __name__ == "__main__":
    main()
