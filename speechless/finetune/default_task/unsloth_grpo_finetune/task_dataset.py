import os, re
from pathlib import Path
from datasets import load_dataset 
from speechless.finetune.dataset_utils.multi_rounds import format_chat
from textwrap import dedent

def generate_prompt_func(example):
    question = example['question']
    answer = example.get('answer', None)

    system_prompt = dedent("""
        Let's think step by step. 
        You are speaking with a professional.
        Reasoning steps: concise, clear.
            - Key points only.
            - No extra words.
            - Max 10 words per step.
            - Math expressions preferred if possible.
        The entire thought process must not exceed 512 words, and the final answer should be no longer than 32 words.
        At the very end of the reply, please strictly provide the answer in the format '$\\boxed{}$.'
        """)
    user_prompt = question

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    prompt = format_chat(messages, add_generation_prompt=True)
    if answer is not None:
        # response = "<think></think>\n\\boxed{" + str(answer) + "}"
        pattern = r"\\boxed{(.+?)}"
        found = re.findall(pattern, answer)
        if found:
            ans = found[-1]
            ans = "$\\boxed{" + ans + "}$."
        else:
            ans = answer.split("</think>")[-1]
            ans = "$\\boxed{" + ans + "}$."
    else:
        raise Exception(f"answer is None")

    return {
        "prompt": prompt,
        "targets": ans,
    }

def load_task_datasets(dataset_path, eval_size: int=200):
    if Path(dataset_path).is_dir():
        dataset = load_dataset(dataset_path, split="train")
    elif os.path.exists(dataset_path):
        # get file name externsion
        _, file_ext = os.path.splitext(dataset_path)
        dataset = load_dataset(file_ext, data_files=dataset_path, split="train")
    else:
        raise FileNotFoundError(f"{dataset_path}")
    print(f"{dataset=}") 

    dataset_columns = dataset.column_names
    dataset = dataset.map(generate_prompt_func)
    dataset = dataset.remove_columns(dataset_columns)
    print(f"final dataset: {dataset}") 

    dataset = dataset.train_test_split(test_size=eval_size)
    train_dataset = dataset['train']
    eval_dataset = dataset['test']
    print(f"{train_dataset=}") 
    print(f"{eval_dataset=}") 

    return train_dataset, eval_dataset
