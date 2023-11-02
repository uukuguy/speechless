"""Example Python client for vllm.entrypoints.api_server"""
# vllm/examples/api_client.py

import argparse
import json
from typing import Iterable, List

import requests


def clear_line(n: int = 1) -> None:
    LINE_UP = '\033[1A'
    LINE_CLEAR = '\x1b[2K'
    for _ in range(n):
        print(LINE_UP, end=LINE_CLEAR, flush=True)


def post_http_request(prompt: str,
                      api_url: str,
                      n: int = 1,
                      stream: bool = False) -> requests.Response:
    headers = {"User-Agent": "Speechless Client"}
    pload = {
        "prompt": prompt,
        "params": {
            "temperature": 0.01,
            "do_sample": True,
            "top_p": 0.9,
            "top_k": 50,
            "num_beams": 1,
            "max_new_tokens": 1024,
        },
    }
    response = requests.post(api_url, headers=headers, json=pload, stream=True)
    return response


def get_streaming_response(response: requests.Response) -> Iterable[List[str]]:
    for chunk in response.iter_lines(chunk_size=8192,
                                     decode_unicode=False,
                                     delimiter=b"\0"):
        print(f"{chunk=}")
        if chunk:
            # data = json.loads(chunk.decode("utf-8"))
            # output = data["text"]
            output = chunk.decode("utf-8")
            yield output


def get_response(response: requests.Response) -> List[str]:
    data = json.loads(response.content)
    # output = data["text"]
    output = data
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=14242)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--prompt", type=str, default="San Francisco is a")
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()
    prompt = args.prompt

    n = args.n
    stream = args.stream

    print(f"Prompt: {prompt!r}\n", flush=True)


    if stream:
        api_url = f"http://{args.host}:{args.port}/agenerate"
        response = post_http_request(prompt, api_url, n, stream)
        num_printed_lines = 0
        full_response = ""
        for h in get_streaming_response(response):
            full_response += h
            print(full_response)
            # clear_line(num_printed_lines)
            # num_printed_lines = 0
            # for i, line in enumerate(h):
            #     num_printed_lines += 1
            #     # print(f"Beam candidate {i}: {line!r}", flush=True)
            #     # print(f"{line\r}", flush=True)
            #     print(line, flush=True)
    else:
        api_url = f"http://{args.host}:{args.port}/generate"
        response = post_http_request(prompt, api_url, n, stream)
        output = get_response(response)
        print(f"{output=}")
        # for i, line in enumerate(output):
        #     print(f"Beam candidate {i}: {line!r}", flush=True)
