#!/usr/bin/env python
"""
This module provides functionality for generating queries using the OpenAI API asynchronously.

The `OpenAICompletion` class is used to generate queries asynchronously using the OpenAI API. The `batch_completion` function
uses `OpenAICompletion` to generate queries in parallel for a list of input rows.

Example usage:
    ```
    from async_openai import batch_completion, CompletionPrompt

    model_name = "gpt-3.5-turbo-0613"
    input_rows = [
        CompletionPrompt(
            prompt_template="The capital of France is ###{city}###.",
            completion_args={"city": "Paris"}
        ),
        CompletionPrompt(
            prompt_template="What is the meaning of life?",
            completion_args={}
        )
    ]
    sampling_params = {
        "temperature": 0.2,
        "max_tokens": 2048,
        "sampling_method": "greedy",
        "num_beams": 1,
        "best_of": 1,
        "stop": "\n\n"
    }
    parallel_threads = 8
    timeout = 30.0

    results = batch_completion(model_name, input_rows, sampling_params, parallel_threads, timeout)
    ```
"""

import time, copy
import random
from typing import Dict, List, Any, Tuple
from tqdm import tqdm
from rich import print
from loguru import logger

from concurrent.futures import ThreadPoolExecutor, as_completed
from func_timeout import FunctionTimedOut, func_timeout

from llm_api import openai_chat_completion, openai_nonchat_completion, titoken_count_tokens

from dataclasses import dataclass, field

@dataclass
class CompletionPrompt:
    """
    A dataclass representing a completion prompt.

    Attributes:
    -----------
    completion_type : str
        The type of completion to use. Can be "completion" or "chat_completion".
    prompt_template : str
        A string containing the prompt template.
    prompt_args : Dict[str, Any]
        A dictionary containing the arguments to use for formatting the prompt template.
    """
    completion_type: str = "completion"
    prompt_template: str = ""
    prompt_args: Dict[str, Any] = field(default_factory=dict)
    sampling_params: Dict[str, Any] = field(default_factory=dict)   
    input_data: Any = None
    result_dicts: List[Dict[str, Any]] = field(default_factory=list)


class OpenAICompletion:
    """
    A class that handles OpenAI completions.

    Attributes:
    -----------
    model_name : str
        The name of the OpenAI model to use for completions.
    timeout : int
        The maximum number of seconds to wait for a completion before timing out.
    sampling_params : Dict[str, Any]
        A dictionary of sampling parameters to use for completions.
    """

    def __init__(self, model_name: str, timeout: int, sampling_params: Dict[str, Any], **kwargs):
        """
        Initializes an instance of the OpenAICompletion class.

        Parameters:
        -----------
        model_name : str
            The name of the OpenAI model to use for completions.
        timeout : int
            The maximum number of seconds to wait for a completion before timing out.
        sampling_params : Dict[str, Any]
            A dictionary of sampling parameters to use for completions.
        """
        self.model_name = model_name
        self.timeout = timeout
        self.sampling_params = sampling_params

        self.err = ""
        self.query = ""
        self.reason = ""
        self.completion = ""

    def do_completion(self, completion_tuple: Tuple) -> dict:
        """
        Generates a completion for the given prompt.

        Parameters:
        -----------
        completion_prompt : CompletionPrompt
            A CompletionPrompt object containing the completion prompt.

        Returns:
        --------
        A dictionary containing the completion, reason, error message, latency, and tokens used.
        """
        start_time = time.time()
        n, completion_prompt = completion_tuple
        self.err = ""
        self.query = ""
        self.reason = ""
        self.completion = ""

        function_to_run = None
        prompt_or_messages = None

        use_chat_completion = completion_prompt.completion_type == 'chat_completion'
        prompt_template = completion_prompt.prompt_template
        prompt_args = completion_prompt.prompt_args
        sampling_params = completion_prompt.sampling_params

        use_chat_completion = False
        if use_chat_completion:
            sys_prompt = prompt_template.split("### Input:")[0]
            user_prompt = prompt_template.split("### Input:")[1].split("### Response:")[0]
            assistant_prompt = prompt_template.split("### Response:")[1]

            if prompt_args:
                user_prompt = user_prompt.format(**prompt_args)
            else:
                user_prompt = user_prompt

            messages = []
            messages.append({"role": "system", "content": sys_prompt})
            messages.append({"role": "user", "content": user_prompt})
            messages.append({"role": "assistant", "content": assistant_prompt})

            function_to_run = openai_chat_completion
            prompt_or_messages = messages
            tokens_used = titoken_count_tokens(model_name=self.model_name, messages=messages)
        else:
            try:
                if prompt_args:
                    prompt = prompt_template.format(**prompt_args)
                else:
                    prompt = prompt_template
            except Exception as e:
                logger.error(f"Error formatting prompt: {e}. {prompt_template=}, {prompt_args=}")
                prompt = prompt_template

            function_to_run = openai_nonchat_completion
            prompt_or_messages = prompt
            tokens_used = titoken_count_tokens(model_name=self.model_name, prompt=prompt)


        try:
            self.completion = func_timeout(
                self.timeout,
                function_to_run,
                args=(
                    self.model_name,
                    prompt_or_messages,
                    sampling_params if sampling_params else self.sampling_params,
                ),
            )

            self.reason = "-"

            print(f"========== prompt_or_messages ==========")
            print(prompt_or_messages)
            print(f"---------- completion ----------")
            print(self.completion)
            print(f"----------------------------------------")

        except FunctionTimedOut:
            logger.error("generating query timed out")
            self.err = "QUERY GENERATION TIMEOUT"
        except Exception as e:
            logger.error(f"Error while generating query: {type(e)}, {e})")
            self.query = ""
            self.reason = ""
            if isinstance(e, KeyError):
                self.err = f"QUERY GENERATION ERROR: {type(e)}, {e}, Completion: {self.completion}"
            else:
                self.err = f"QUERY GENERATION ERROR: {type(e)}, {e}"

        return {
            "n": n,
            "completion": self.completion,
            "reason": self.reason,
            "err": self.err,
            "latency_seconds": time.time() - start_time,
            "tokens_used": tokens_used,
        }


def batch_completion(model_name: str, 
                     input_completion_prompts: List[CompletionPrompt], 
                     sampling_params: Dict[str, Any], 
                     parallel_threads: int = 1, 
                     timeout: float = 30.0,
                     completion_limit: int = 1,
                     ):
    """
    Asynchronously generates completions for a batch of prompts using OpenAI's API.

    Args:
        model_name (str): The name of the OpenAI model to use for the completions.
        input_completion_prompts (List[CompletionPrompt]): A list of CompletionPrompt, where each dictionary represents a prompt to generate a completion for. Each dictionary should have the following keys:
            - 'prompt_template': A string representing the prompt template to use for the completion.
            - 'completion_args': A dictionary representing the arguments to use for the completion.
        sampling_params (Dict[str, Any]): A dictionary representing the sampling parameters to use for the completions. See the OpenAI API documentation for more information.
        parallel_threads (int, optional): The number of threads to use for generating the completions. Defaults to 1.
        timeout (float, optional): The maximum amount of time (in seconds) to wait for a completion to be generated. Defaults to 30.0.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, where each dictionary represents the results of generating a completion for a prompt. Each dictionary will have the same keys as the input dictionary, as well as the following additional keys:
            - 'latency_seconds': The amount of time (in seconds) it took to generate the completion.
            - 'tokens_used': The number of tokens used to generate the completion.
            - 'error_query_gen': A boolean indicating whether there was an error generating the completion.
            - 'timeout': A boolean indicating whether the completion timed out.
    """
    # output_completion_prompts = []

    real_completion_prompts = []    
    for n, completion_prompt in enumerate(input_completion_prompts):
        for i in range(completion_limit):
            real_completion_prompt = copy.deepcopy(completion_prompt)
            real_completion_prompts.append((n, real_completion_prompt))
    real_completion_prompts = random.sample(real_completion_prompts, len(real_completion_prompts))

    with ThreadPoolExecutor(parallel_threads) as executor:
        # for each row, generate a query using the generator asynchronously
        futures = []
        for n, completion_prompt in real_completion_prompts:

            openai_completion = OpenAICompletion(
                model_name=model_name,
                timeout=timeout,
                sampling_params=sampling_params,
            )

            futures.append(
                executor.submit(
                    openai_completion.do_completion, 
                    completion_tuple=(n, completion_prompt),
                    )
                )

        # wait for all the queries to finish
        # for each query, save the results into the output dataframe
        total_tried = 0
        total_correct = 0
        for f in (pbar := tqdm(as_completed(futures), total=len(futures))):
            total_tried += 1
            i = futures.index(f)
            i_n = real_completion_prompts[i][0]
            completion_prompt = input_completion_prompts[i_n]
            result_dict = f.result()
            # print(f"{completion_prompt.prompt_template=}, {completion_prompt.prompt_args=}")
            # print(f"{result_dict=}")

            # # query_gen = result_dict["query"]
            # reason = result_dict["reason"]
            # err = result_dict["err"]

            # # save custom metrics
            # if "latency_seconds" in result_dict:
            #     row["latency_seconds"] = result_dict["latency_seconds"]
            # if "tokens_used" in result_dict:
            #     row["tokens_used"] = result_dict["tokens_used"]

            # # save failures into relevant columns in the dataframe
            # if "GENERATION ERROR" in err:
            #     row["error_query_gen"] = 1
            # elif "TIMEOUT" in err:
            #     row["timeout"] = 1
            # else:
            #     pass
            #     # total_correct += 1
            completion_prompt.result_dicts.append(result_dict)

            # output_completion_prompts.append((i_n, completion_prompt))
            # pbar.set_description(
            #     f"Correct so far: {total_correct}/{total_tried} ({100*total_correct/total_tried:.2f}%)"
            # )

    return input_completion_prompts

    # Sort the output completion prompts by their original index
    # output_completion_prompts = sorted(output_completion_prompts, key=lambda x: x[0])
    # output_completion_prompts = [completion_prompt for _, completion_prompt in output_completion_prompts]

    # return output_completion_prompts


def default_argument_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", type=str, help="The name of the OpenAI model to use for the completions.")
    parser.add_argument("-p", "--parallel_threads", type=int, default=8, help="The number of threads to use for generating the completions.")
    parser.add_argument("-t", "--timeout", type=float, default=30.0, 
                        help="The maximum amount of time (in seconds) to wait for a completion to be generated.")

    parser.add_argument("--temperature", type=float, default=0.2, help="The temperature for the GPT-3 API")
    parser.add_argument("--max_tokens", type=int, default=2048, help="The max tokens for the GPT-3 API")
    parser.add_argument("--sampling_method", type=str, default="greedy", choices=['greedy', 'beam_search', 'normal'], 
                        help="The sampling method for the GPT-3 API")
    parser.add_argument("--num_beams", type=int, default=1, help="The number of beams for the GPT-3 API")
    parser.add_argument("--top_p", type=float, default=1.0, help="The number of beams for the GPT-3 API")
    parser.add_argument("--top_k", type=int, default=-1, help="The number of beams for the GPT-3 API")
    parser.add_argument("--best_of", type=int, default=1, help="The number of beams for the GPT-3 API")
    parser.add_argument("--stop", type=str, default=None, help="The stop token for the GPT-3 API")
    parser.add_argument("--completion_limit", type=int, default=1)


    return parser

def sampling_params_from_args(args):
    sampling_params={
        'temperature': args.temperature,
        'top_p': args.top_p,
        'top_k': args.top_k,
        'max_tokens': args.max_tokens,
        'sampling_method': args.sampling_method,
        'n': args.num_beams,
        'best_of': args.best_of,
        'stop': args.stop.split(",") if args.stop else [],
    }

    return sampling_params

def get_args():
    parser = default_argument_parser()

    parser.add_argument("-f", "--prompt_file", type=str, required=True, help="The path to the file containing the prompt template.")

    args = parser.parse_args()
    return args


def build_completion_prompts(dataset, prompt_template, sampling_params):
    completion_prompts = []
    for data in dataset:
        s_params = copy.deepcopy(sampling_params)
        s_params['stop'] = data['stop_tokens']

        completion_prompt = CompletionPrompt(
            completion_type="commpletion",
            prompt_template=data['prompt'],
            prompt_args={},
            input_data=data,
            sampling_params=s_params,
        ) 
        completion_prompts.append(completion_prompt)
    
    return completion_prompts


def main():
    args = get_args()
    print(f"{args=}")

    prompt_template = ""
    with open(args.prompt_file) as file:
        prompt_template = file.read()

    from datasets import load_dataset
    lang_dataset = load_dataset("nuprl/MultiPL-E", "humaneval-py", split="test")

    sampling_params = sampling_params_from_args(args)
    input_completion_prompts = build_completion_prompts(lang_dataset, prompt_template=prompt_template, sampling_params=sampling_params)

    output_completion_prompts = batch_completion(
        model_name=args.model_name,
        input_completion_prompts=input_completion_prompts,
        sampling_params=sampling_params,
        parallel_threads=args.parallel_threads,
        timeout=args.timeout,
        )

if __name__ == '__main__':
    main()