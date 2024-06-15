# Adapted from https://github.com/lm-sys/FastChat/blob/b3c8bd71637d6c88206a360be436e7941b4fffb4/fastchat/eval/qa_baseline_gpt35.py
"""Generate answers with GPT-3.5"""
# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
# speechless.infer.infer.py

import os, json
import time
from tqdm import tqdm
import concurrent.futures
from typing import List, Tuple
from abc import ABC, abstractmethod
import torch

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

    def generate_prompt(self, instruction: str, input: str=""):
        prompt_template = self.PROMPT_DICT["prompt_input"] if input else self.PROMPT_DICT["prompt_no_input"]
        prompt = prompt_template.format(instruction=instruction, input=input)
        return prompt


class AIModel(ABC):
    def __init__(self, model_name_or_path=None, gen_args={}):
        self.model_name_or_path = model_name_or_path
        self.gen_args = gen_args
        self.model = self.load_model()

    @abstractmethod
    def load_model(self, *args, **kwargs):
        """Loads a model, that will be responsible for scoring.

        Returns:
            A model object
        """
        pass


    @abstractmethod
    def generate(self, *args, **kwargs) -> str:
        """Runs the model to output LLM response.

        Returns:
            A string.
        """
        pass


class OpenAIModel(AIModel):
    def __init__(self, model_name_or_path=None, max_tokens=2048):
        super().__init__(model_name_or_path, max_tokens)

    def load_model(self):
        from openai import OpenAI
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "sk-unknown"))

    def generate(self, prompt: str, max_tokens) -> str:
        for _ in range(3):
            try:
                response = self.client.completions.create(model=self.model_name_or_path,
                # prompt=prompt,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                max_tokens=max_tokens)
                generated_text = response["choices"][0]["text"]

            except Exception as e:
                print("[ERROR]", e)
                generated_text = "#ERROR#"
                time.sleep(1)

        return generated_text


from .llm import VllmAIModel as VllmModel
# class VllmModel(AIModel):
#     def __init__(self, model_name_or_path=None, max_tokens=2048):
#         super().__init__(model_name_or_path, max_tokens)

#     def load_model(self):
#         from vllm import LLM

#         return LLM(model=self.model_path, trust_remote_code=True, tensor_parallel_size=torch.cuda.device_count())

#     def generate(self, prompt: str) -> str:
#         from vllm import SamplingParams

#         sampling_params = SamplingParams(**self.gen_args)
#         outputs = self.model.generate(prompt, sampling_params)
#         generated_text = ''
#         for output in outputs:
#             generated_text += output.outputs[0].text

#         return generated_text

class LlamaCppModel(AIModel):
    def __init__(self, model_path=None, n_gpu_layers=32, n_ctx=2048, max_tokens=500, *args, **kwargs):
        self.model_name = 'llama-cpp'
        self.model_path = model_path
        self.n_gpu_layers = n_gpu_layers
        self.max_tokens = max_tokens
        self.n_ctx = n_ctx
        self.chat_model = self.load_model()


    def load_model(self):
        from llama_cpp import Llama
        return Llama(model_path=self.model_path, n_gpu_layers=self.n_gpu_layers, n_ctx=self.n_ctx)


    def generate(self, prompt: str) -> str:
        output = self.chat_model(prompt, max_tokens=self.max_tokens)
        output = output['choices'][0]['text'].strip()
        return output

def get_response_from_llm(ai_model: AIModel, qid: int, instruction: str, max_tokens: int, prompt_type: str="raw"):
    assert ai_model is not None, "Model is not loaded"

    generated_text = ai_model.generate(instruction, max_tokens=max_tokens)
    ans = {
        'qid': qid,
        'text': generated_text
    }

    return ans

def concurrent_generate(llm: str, model_name: str, instructions: List[Tuple[str, str]], parallel: int=8, max_tokens: int=4096, prompt_type: str="raw"):
    if llm == "vllm":
        model = VllmModel(model_name)
    elif llm == "llamacpp":
        model = LlamaCppModel(model_name)
    elif llm == "openai":
        model = OpenAIModel(model_name)
    else:
        raise ValueError(f"Unsupported LLM: {llm}")

    responses = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = []
        for qid, instruction in instructions.items():
            if prompt_type == "raw":
                prompt = instruction
            elif prompt_type == "alpaca":
                prompter = AlpacaPrompter()
                prompt = prompter.generate_prompt(instruction, "")

            future = executor.submit(get_response_from_llm, model_name, qid, prompt, max_tokens)
            futures.append(future)

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            a = (future.result())
            # print(f"{a=}")
            responses.append(a)

    return responses


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ChatGPT answer generation.")
    parser.add_argument("-l", "--llm", choices=["openai", "vllm", "llamacpp", "litellm", "ollama"], default="vllm")
    parser.add_argument("-m", "--model_name", type=str, default="gpt-3.5-turbo")
    parser.add_argument("-q", "--questions_file", required=True, type=str)
    parser.add_argument("-o", "--output_file", type=str, default="answers.jsonl")
    parser.add_argument("-p", "--parallel", type=int, default=8)
    parser.add_argument( "--max_tokens", type=int, default=4096, help="maximum number of tokens produced in the output")
    parser.add_argument("--prompt_type", default="raw", choices=['raw', 'alpaca'])
    args = parser.parse_args()

    prompter = AlpacaPrompter()
    
    instructions = []
    with open(os.path.expanduser(args.questions_file)) as f:
        for q_id, line in enumerate(f):
            if not line:
                continue
            q = json.loads(line)
            instructions.append((q_id, q["input"]))

    responses = concurrent_generate(args.model_name, instructions, args.parallel, args.max_tokens, args.prompt_type)

    responses.sort(key=lambda x: x["qid"])

    # with open(os.path.expanduser(args.output_file), "w") as f:
    #     for q, a in tqdm(zip(questions, answers), total=len(questions)):
    #         q["target"] = a["text"]
    #         f.write(f"{json.dumps(q, ensure_ascii=False)}\n")
    # print(f"Saved {len(answers)} answers to {args.output_file}")
