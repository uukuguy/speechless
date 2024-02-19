#!/usr/bin/env python
"""
Usage:  python -m speechless.generate.wasmedge --model_path sus-chat-34b.Q8_0.gguf --verbose --max_tokens 512 --prompt_file $SPEECHLESS_ROOT/speechless/generate/prompts/hello_llm_en.txt
"""

import os


def generate(args):
    if args.prompt_file:
        prompt = open(args.prompt_file).read().strip()
    else:
        prompt = args.prompt
    # if not args.ignore_chat_template and (
    #     hasattr(tokenizer, "apply_chat_template")
    #     and tokenizer.chat_template is not None
    # ):
    #     messages = [{"role": "user", "content": prompt}]
    #     prompt = tokenizer.apply_chat_template(
    #         messages, tokenize=False, add_generation_prompt=True
    #     )

    """
    wasmedge --dir .:. \
        --nn-preload default:GGML:AUTO:${MODEL_PATH} \
        ./llama-chat.wasm \
        -p mistral-instruct \
    """

    GENERATE_OPTS = f"--temp {args.temperature} --top-p {args.top_p} --repeat-penalty {args.repeat_penalty}"
    cmd = f"wasmedge --dir .:. --nn-preload default:GGML:AUTO:{args.model_path} llama-chat.wasm -p {args.prompt_type} -g {args.n_gpu_layers} --ctx-size {args.ctx_size} {GENERATE_OPTS}"
    # print("Running command: ", cmd)

    os.system(cmd)

def run_api_server(args):
    """
    wasmedge --dir .:. \
    --nn-preload default:GGML:AUTO:${MODEL_PATH} \
    ./llama-api-server.wasm \
    -p mistral-instruct \
    --socket-addr 127.0.0.1:16384 --log-prompts --log-stat
    """
    GENERATE_OPTS = f"--temp {args.temperature} --top-p {args.top_p} --repeat-penalty {args.repeat_penalty}"
    model_name = ".".join(args.model_path.split("/")[-1].split(".")[:-1])
    cmd = f"wasmedge --dir .:. --nn-preload default:GGML:AUTO:{args.model_path} llama-api-server.wasm -p {args.prompt_type} -g {args.n_gpu_layers} --ctx-size {args.ctx_size} {GENERATE_OPTS} --socket-addr 127.0.0.1:{args.port} --log-prompts --log-stat --web-ui {args.web_ui} --model-name {model_name}"
    # print("Running command: ", cmd)

    os.system(cmd)
    

available_prompt_types = ['llama-2-chat', 'codellama-instruct', 'codellama-super-instruct', 'mistral-instruct-v0.1', 'mistral-instruct', 'mistrallite', 'openchat', 'human-assistant', 'vicuna-1.0-chat', 'vicuna-1.1-chat', 'chatml', 'baichuan-2', 'wizard-coder', 'zephyr', 'stablelm-zephyr', 'intel-neural', 'deepseek-chat', 'deepseek-coder', 'solar-instruct']

def get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--trust-remote-code", action="store_true", help="Enable trusting remote code for tokenizer")
    parser.add_argument("--prompt_type", type=str, choices=['llama-2-chat', 'chatml', 'mistral-instruct'], help="prompt type")

    parser.add_argument("--api_server", action="store_true", help="run api server")
    parser.add_argument("--port", type=int, default=16384, help="port to run the server on")
    parser.add_argument("--web_ui", type=str, default = "chatbot-ui", help="web ui path")

    parser.add_argument("--model_path", type=str, required=True, help="GGUF file")
    parser.add_argument("--adapter_file", type=str, help="adapter file path")
    parser.add_argument("--prompt", type=str, help="prompt to run")
    parser.add_argument("--prompt_file", type=str, help="prompt file")

    parser.add_argument("--temperature", type=float, default=0.7, help="temperature")
    parser.add_argument("--max_tokens", type=int, default=512, help="max tokens")
    parser.add_argument("--ctx_size", type=int, default=512, help="context size")

    parser.add_argument("--top_p", type=float, default=0.9, help="top p")
    parser.add_argument("--top_k", type=int, default=40, help="top k")
    parser.add_argument("--repeat_penalty", type=float, default=1.1, help="repeat penalty")

    parser.add_argument("--n_gpu_layers", type=int, default=512, help="number of layers on GPU")

    parser.add_argument("--verbose", action="store_true", help="verbose")
    parser.add_argument("--seed", type=int, default=0, help="seed")

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    if args.api_server:
        run_api_server(args)
    else:
        generate(args)


if __name__ == '__main__':
    main()
