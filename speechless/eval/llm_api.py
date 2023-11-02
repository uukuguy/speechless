import time
from typing import Dict, List, Any
import openai, tiktoken
from loguru import logger


def openai_chat_completion(model_name: str, messages: List[str], sampling_params: Dict[str, Any]) -> str:
    """
    This function generates a chat response using the OpenAI API.
    model_name: the name of the OpenAI model to use for the chat response.
    messages: a list of messages to use as context for the chat response.
    sampling_params: a dictionary of sampling parameters to use for the chat response.
    Returns the generated chat response as a string.
    """
    generated_text = ""
    try:
        completion = openai.ChatCompletion.create(
            model=model_name,
            messages=messages,
            **sampling_params,
        )
        generated_text = completion["choices"][0]["message"]["content"]
        print(f"{generated_text=}")
    except (openai.error.RateLimitError, openai.error.ServiceUnavailableError) as e:
        logger.warning("Model overloaded. Pausing for 5s before retrying...")
        time.sleep(5)
        # Retry the api call after 5s
        completion = openai.ChatCompletion.create(
            model=model_name,
            messages=messages,
            **sampling_params,
        )
        generated_text = completion["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(type(e), e)
    return generated_text


def openai_nonchat_completion(model_name: str, prompt: str, sampling_params: Dict[str, Any]) -> str:
    """
    This function generates a non-chat response using the OpenAI API.
    model_name: the name of the OpenAI model to use for the non-chat response.
    prompt: the prompt to use for the non-chat response.
    sampling_params: a dictionary of sampling parameters to use for the non-chat response.
    Returns the generated non-chat response as a string.
    """
    generated_text = ""
    try:
        completion = openai.Completion.create(
            model=model_name,
            prompt=prompt,
            **sampling_params,
        )
        generated_text = completion["choices"][0]["text"]
        logger.info(f"[nonchat] {generated_text=}")
    except (openai.error.RateLimitError, openai.error.ServiceUnavailableError) as e:
        logger.warning("Model overloaded. Pausing for 5s before retrying...")
        time.sleep(5)
        # Retry the api call after 5s
        completion = openai.ChatCompletion.create(
            model=model_name,
            prompt=prompt,
            **sampling_params,
        )
        generated_text = completion["choices"][0]["text"]
    except Exception as e:
        logger.error(type(e), e)
    return generated_text


def titoken_count_tokens(model_name: str = "gpt-3.5-turbo-0613", messages: List[Dict[str, str]] = [], prompt: str = "") -> int:
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