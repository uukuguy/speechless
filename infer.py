# Adapted from https://github.com/lm-sys/FastChat/blob/b3c8bd71637d6c88206a360be436e7941b4fffb4/fastchat/eval/qa_baseline_gpt35.py
"""Generate answers with GPT-3.5"""
# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
import argparse
import json
import os
import time
import concurrent.futures

import openai
from tqdm import tqdm
import shortuuid
openai.api_key = os.getenv("OPENAI_API_KEY", "sk-unknown")

def get_answer(model_name, question_id: int, prompt: str, max_tokens: int):
    ans = {
        "answer_id": shortuuid.uuid(),
        "question_id": question_id,
    }
    for _ in range(3):
        try:
            response = openai.Completion.create(
                model=model_name,
                prompt=prompt,
                # messages=[
                #     {"role": "system", "content": "You are a helpful assistant."},
                #     {
                #         "role": "user",
                #         "content": question,
                #     },
                # ],
                max_tokens=max_tokens,
            )
            # ans["text"] = response["choices"][0]["message"]["content"]
            ans["text"] = response["choices"][0]["text"]
            return ans
        except Exception as e:
            print("[ERROR]", e)
            ans["text"] = "#ERROR#"
            time.sleep(1)
    return ans


class AlpacaPrompter:
    PROMPT_DICT = {
        "prompt_input": (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
        ),
        "prompt_no_input": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:\n"
        ),
    }

    def generate(self, instruction: str, input: str):
        prompt_template = self.PROMPT_DICT["prompt_input"] if input else self.PROMPT_DICT["prompt_no_input"]
        prompt = prompt_template.format(instruction=instruction, input=input)
        return prompt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChatGPT answer generation.")
    parser.add_argument("-m", "--model_name", type=str, default="gpt-3.5-turbo")
    parser.add_argument("-q", "--questions_file", required=True, type=str)
    parser.add_argument("-o", "--output_file", type=str, default="answers.jsonl")
    parser.add_argument("-p", "--parallel", type=int, default=8)
    parser.add_argument( "--max_tokens", type=int, default=4096, help="maximum number of tokens produced in the output")
    args = parser.parse_args()

    prompter = AlpacaPrompter()
    
    questions = []
    questions_dict = {}
    with open(os.path.expanduser(args.questions_file)) as f:
        for q_id, line in enumerate(f):
            if not line:
                continue
            q = json.loads(line)
            questions.append(q)
            instruction = q["input"]
            input = ""
            # instruction = q["instruction"]
            # input = q.get("input", "")
            prompt = prompter.generate(instruction, input)
            questions_dict[q_id] = prompt

    answers = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = []
        for qid, question in questions_dict.items():
            future = executor.submit(get_answer, args.model_name, qid, question, args.max_tokens)
            futures.append(future)

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            a = (future.result())
            print(f"{a=}")
            answers.append(a)

    answers.sort(key=lambda x: x["question_id"])

    with open(os.path.expanduser(args.output_file), "w") as f:
        for q, a in tqdm(zip(questions, answers), total=len(questions)):
            q["target"] = a["text"]
            f.write(f"{json.dumps(q, ensure_ascii=False)}\n")
    print(f"Saved {len(answers)} answers to {args.output_file}")
