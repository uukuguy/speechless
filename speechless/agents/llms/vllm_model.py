#!/usr/bin/env python

import time
from termcolor import colored
from typing import Optional, List, Dict
import torch
from typing import Optional
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from ..utils import process_system_message
from .model_adapter import get_conversation_template
from .utils import SimpleChatIO, generate_stream, react_parser

import uuid
def random_uuid() -> str:
    return str(uuid.uuid4().hex)

import os
os.environ['RAY_memory_monitor_refresh_ms'] = '0'

import tiktoken
def count_tokens(model: str = None, messages: List[Dict[str, str]] = [], text: str = "") -> int:
    """
    This function counts the number of tokens used in a prompt
    model: the model used to generate the prompt. can be one of the following: gpt-3.5-turbo-0613, gpt-4-0613, text-davinci-003
    messages: (only for OpenAI chat models) a list of messages to be used as a prompt. Each message is a dict with two keys: role and content
    prompt: (only for text-davinci-003 model) a string to be used as a prompt
    """
    if model is None:
        tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo-0613")
    else:
        tokenizer = tiktoken.encoding_for_model(model)
    num_tokens = 0
    if messages:
        for message in messages:
            for _, value in message.items():
                num_tokens += len(tokenizer.encode(value))
    else:
        num_tokens = len(tokenizer.encode(text))
    return num_tokens

from vllm import LLM, EngineArgs, LLMEngine, SamplingParams
class VllmEngine:
    def __init__(self, model_name_or_path: str):
        self.model_name = model_name_or_path
        self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo-0613")
        num_gpus = torch.cuda.device_count()
        self.engine = self._create_engine(
            model_name_or_path=model_name_or_path, 
            tensor_parallel_size=num_gpus, 
            trust_remote_code=True,
            )

    def _create_engine(self, model_name_or_path, tensor_parallel_size=1, trust_remote_code=True):
        if tensor_parallel_size < 1:
            tensor_parallel_size = torch.cuda.device_count()

        params = dict(
            disable_log_stats = True,

            model = model_name_or_path,
            tensor_parallel_size = tensor_parallel_size,
            trust_remote_code = trust_remote_code,
            # download_dir: Optional[str] = None
            # load_format: str = 'auto'
            dtype = 'bfloat16', #'float16' if is_gptq or is_awq else 'bfloat16',
            # seed: int = 0
            # max_model_len: Optional[int] = None
            worker_use_ray = True,
            # pipeline_parallel_size = num_gpus,

            # KV cache arguments
            # block_size: int = 16
            # swap_space: int = 4  # GiB CPU swap space size (GiB) per GPU
            # gpu_memory_utilization: float = 0.90
            # max_num_batched_tokens: int = 2560 # maximum number of batched tokens per iteration
            # max_num_seqs: int = 256 # maximum number of sequences per iteration
            # disable_log_stats: bool = False
            # revision: Optional[str] = None

            # quantization = 'awq' if is_awq else None # ['awq', Non] Method used to quantize the weights
        )
        engine_args = EngineArgs(**params)

        engine = LLMEngine.from_engine_args(engine_args)

        return engine

    def generate(self, prompt: str, sampling_params: dict = None):
        generated_text = ""

        request_id = random_uuid()
        request_id = random_uuid()
        if sampling_params is None:
            sampling_params = self.sample_params

        vllm_sampling_params = SamplingParams(**sampling_params)
        self.engine.add_request(request_id, prompt, vllm_sampling_params)
        while True:
            request_outputs = self.engine.step()
            request_output = request_outputs[0]
            # print(f"{type(request_output)=}")
            # print(f"{request_output=}")
            if request_output.finished:
                generated_text = request_output.outputs[0].text
            # for request_output in request_outputs:
            #     if request_output.finished:
            #         print(request_output)

            if not self.engine.has_unfinished_requests():
                break

        return {"text": generated_text}


class VllmModel:
    def __init__(
            self, 
            model_name_or_path: str, 
            template:str="tool-llama-single-round", 
            max_sequence_length: int=8192,
            sampling_params: dict = None,
        ) -> None:
        super().__init__()
        self.model_name = model_name_or_path
        self.template = template
        self.max_sequence_length = max_sequence_length
        defaul_sampling_params = {
            'temperature': 0.5,
            'max_tokens': 512,
            "top_p": 1.0,
            "top_k": 50,
            # "repetition_penalty": 1.2,
        }
        self.sampling_params = sampling_params if sampling_params is not None else defaul_sampling_params

        self.vllm_engine = VllmEngine(model_name_or_path)

    # def prediction_old(self, prompt: str, stop: Optional[List[str]] = None) -> str:
    #     with torch.no_grad():
    #         gen_params = {
    #             "model": "",
    #             "prompt": prompt,
    #             "temperature": 0.5,
    #             "max_new_tokens": 512,
    #             "stop": "</s>",
    #             "stop_token_ids": None,
    #             "echo": False
    #         }
    #         generate_stream_func = generate_stream
    #         output_stream = generate_stream_func(self.model, self.tokenizer, gen_params, "cuda", self.max_sequence_length, force_generate=True)
    #         outputs = self.chatio.return_output(output_stream)
    #         prediction = outputs.strip()
    #     return prediction

    # def prediction_hf(self, prompt: str, stop: Optional[List[str]] = None) -> str:
    #     with torch.no_grad():
    #         # gen_params = {
    #         #     "model": "",
    #         #     "prompt": prompt,
    #         #     "temperature": 0.5,
    #         #     "max_new_tokens": 512,
    #         #     "stop": "</s>",
    #         #     "stop_token_ids": None,
    #         #     "echo": False
    #         # }
    #         # generate_stream_func = generate_stream
    #         # output_stream = generate_stream_func(self.model, self.tokenizer, gen_params, "cuda", self.max_sequence_length, force_generate=True)
    #         # outputs = self.chatio.return_output(output_stream)
    #         # prediction = outputs.strip()

    #         input_ids = self.tokenizer(
    #             prompt,
    #             return_token_type_ids=False,
    #             return_attention_mask=True,
    #             return_tensors="pt",
    #         ).to("cuda")
    #         sampling_params = {
    #             'do_sample': True,
    #             'temperature': 0.5,
    #             'max_new_tokens': 2048,
    #             "top_p": 1.0,
    #             "top_k": 50,
    #             "repetition_penalty": 1.2,
    #         }
    #         gen_tokens = self.model.generate(**input_ids, pad_token_id=self.tokenizer.pad_token_id, **sampling_params)
    #         result = self.tokenizer.batch_decode(
    #             gen_tokens[:, input_ids["input_ids"].shape[1] :], skip_special_tokens=True
    #         )
    #         generated_text = result[0]
    #         print(f">>>>> {generated_text=}")

    #     return generated_text

    def prediction(self, prompt: str, stop: Optional[List[str]] = None) -> str:
    # def prediction_vllm(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        sampling_params = {
            'temperature': 0.5,
            'max_tokens': 2048,
            "top_p": 1.0,
            "top_k": 50,
            # "repetition_penalty": 1.2,
        }
        generated_text = self.vllm_engine.generate(prompt, sampling_params)["text"]
        # print(f">>>>> {generated_text=}")

        return generated_text
        
    def add_message(self, message):
        self.conversation_history.append(message)

    def change_messages(self,messages):
        self.conversation_history = messages

    def display_conversation(self, detailed=False):
        role_to_color = {
            "system": "red",
            "user": "green",
            "assistant": "blue",
            "function": "magenta",
        }
        print("before_print"+"*"*50)
        for message in self.conversation_history:
            print_obj = f"{message['role']}: {message['content']} "
            if "function_call" in message.keys():
                print_obj = print_obj + f"function_call: {message['function_call']}"
            print_obj += ""
            print(
                colored(
                    print_obj,
                    role_to_color[message["role"]],
                )
            )
        print("end_print"+"*"*50)

    def parse(self, functions, process_id, **args):
        conv = get_conversation_template(self.template)
        if self.template == "tool-llama":
            roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
        elif self.template == "tool-llama-single-round" or self.template == "tool-llama-multi-rounds":
            roles = {"system": conv.roles[0], "user": conv.roles[1], "function": conv.roles[2], "assistant": conv.roles[3]}

        self.time = time.time()
        conversation_history = self.conversation_history
        prompt = ''
        for message in conversation_history:
            role = roles[message['role']]
            content = message['content']
            if role == "System" and functions != []:
                content = process_system_message(content, functions)
            prompt += f"{role}: {content}\n"
        prompt += "Assistant:\n"
        
        if functions != []:
            predictions = self.prediction(prompt)
        else:
            predictions = self.prediction(prompt)

        # decoded_token_len = len(self.tokenizer(predictions))
        # if process_id == 0:
        #     print(f"[process({process_id})]total tokens: {decoded_token_len}")
        decoded_token_len = count_tokens(text=predictions)

        # react format prediction
        thought, action, action_input = react_parser(predictions)
        message = {
            "role": "assistant",
            "content": thought,
            "function_call": {
                "name": action,
                "arguments": action_input
            }
        }
        return message, 0, decoded_token_len


if __name__ == "__main__":
    # can accept all huggingface LlamaModel family
    llm = ToolLLaMA("decapoda-research/llama-7b-hf")
    messages = [
        {'role': 'system', 'content': '''You are AutoGPT, you can use many tools(functions) to do
the following task.\nFirst I will give you the task description, and your task start.\nAt each step, you need to give your thought to analyze the status now and what to do next, with a function call to actually excute your step.\nAfter the call, you will get the call result, and you are now in a new state.\nThen you will analyze your status now, then decide what to do next...\nAfter many (Thought-call) pairs, you finally perform the task, then you can give your finial answer.\nRemember: \n1.the state change is , you can\'t go
back to the former state, if you want to restart the task, say "I give up and restart".\n2.All the thought is short, at most in 5 sentence.\nLet\'s Begin!\nTask description: Use numbers and basic arithmetic operations (+ - * /) to obtain exactly one number=24. Each
step, you are only allowed to choose two of the left numbers to obtain a new number. For example, you can combine [3,13,9,7] as 7*9 - 3*13 = 24.\nRemember:\n1.all of the number must be used , and must be used ONCE. So Only when left numbers is exact 24, you will win. So you don\'t succeed when left number = [24, 5]. You succeed when left number = [24]. \n2.all the try takes exactly 3 steps, look
at the input format'''}, 
{'role': 'user', 'content': '\nThe real task input is: [1, 2, 4, 7]\nBegin!\n'}
]
    functions = [{'name': 'play_24', 'description': '''make your current conbine with the format "x operation y = z (left: aaa) " like "1+2=3, (left: 3 5 7)", then I will tell you whether you win. This is the ONLY way
to interact with the game, and the total process of a input use 3 steps of call, each step you can only combine 2 of the left numbers, so the count of left numbers decrease from 4 to 1''','parameters':{'type': 'object', 'properties':{}}}]#, 'parameters': {'type': 'object', 'properties': {'input': {'type': 'string', 'description': 'describe what number you want to conbine, and how to conbine.'}}, 'required': ['input']}}]

    llm.change_messages(messages)
    output = llm.parse(functions=functions)
    print(output)