#!/usr/bin/env python

import os
from openai import OpenAI

import time
import sys
import asyncio
from loguru import logger
from rich.console import Console
from rich.markdown import Markdown
console = Console()

def print_markdown(text):
    console.print(Markdown(text))

import tiktoken
from typing import List, Dict
def tiktoken_count_tokens(model_name: str = "gpt-3.5-turbo-0613", messages: List[Dict[str, str]] = [], prompt: str = "") -> int:
    """
    This function counts the number of tokens used in a prompt.
    model_name: the model used to generate the prompt. can be one of the following: gpt-3.5-turbo-0613, gpt-4-0613, text-davinci-003
    messages: (only for OpenAI chat models) a list of messages to be used as a prompt. Each message is a dict with two keys: role and content
    prompt: (only for text-davinci-003 model) a string to be used as a prompt
    Returns the number of tokens used in the prompt as an integer.
    """
    gpt_models = ['gpt-3.5-turbo-0613', 'gpt-4-0613', 'text-davinci-003']
    if model_name in gpt_models:
        tokenizer = tiktoken.encoding_for_model(model_name)
    else:
        tokenizer = tiktoken.encoding_for_model('gpt-3.5-turbo-0613')

    num_tokens = 0
    if messages:
        for message in messages:
            for _, value in message.items():
                num_tokens += len(tokenizer.encode(value))
    else:
        num_tokens = len(tokenizer.encode(prompt))

    return num_tokens

OPENAI_API_KEY=os.getenv("OPENAI_API_KEY"), 
OPENAI_BASE_URL=os.getenv("OPENAI_BASE_URL")
OPENAI_DEFAULT_MODEL=os.getenv("OPENAI_DEFAULT_MODEL")

logger.debug(f"{OPENAI_BASE_URL=}")
logger.debug(f"{OPENAI_API_KEY=}")
logger.debug(f"{OPENAI_DEFAULT_MODEL=}")

client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
)

# async def get_response(prompt):
def get_response(prompt, stream: bool = True, verbose: bool = False):
    if verbose:
        logger.info(f"Prompt: {PROMPT}")
    try:
        response = client.chat.completions.create(
            model=OPENAI_DEFAULT_MODEL,
            messages=[
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': PROMPT}
                ],
            temperature=0.6,
            stream=stream,
        )
    except Exception as e:
        logger.error(f"Error: {e}")
        return ""

    if verbose:
        logger.info(f"{response=}")

    generated_text = ""
    if stream:
        # async for chunk in response:
        for chunk in response:
            chunk_content = chunk.choices[0].delta.content
            if chunk_content is not None:
                print(chunk_content, end="")
                generated_text += chunk_content
    else:
        generated_text = response.choices[0].message.content
    return generated_text

start_time = time.time()

PROMPT=sys.argv[1] if len(sys.argv) > 1 else "Which is bigger, 9.9 or 9.11?"
# generated_text = asyncio.run(get_response(PROMPT))
generated_text = get_response(PROMPT, stream=True, verbose=True)
end_time = time.time()

print(f"\n\n---------- Final Response ----------")
print_markdown(generated_text)
print(f"\n--------------------")

logger.info(f"Time taken: {end_time - start_time}")
num_tokens = tiktoken_count_tokens(prompt=generated_text)
logger.info(f"Number of tokens used: {num_tokens}")
logger.info(f"Averaged tokens per second: {num_tokens / (end_time - start_time):.2f} tps.")