"""
Usage: python -m speechless.infer.openrouter --stream --prompt "Give me quick sort python code"
Parameters:
    --api_url: OpenRouter API URL # default: https://openrouter.ai/api/v1
    --api_key: OpenRouter API Key # default: $OPENROUTER_API_KEY
    --model: Model name # default: huggingfaceh4/zephyr-7b-beta
    --stream: Stream or not # default: False
    --site_url: Your site URL 
    --app_name: Your app name
    --max_tokens: Max tokens # default: 4096
    --temperature: Temperature # default: 0.2
    --prompt: Prompt
"""
import os, sys
from openai import OpenAI

# gets API Key from environment variable OPENAI_API_KEY

# ----- $0 -----
# MODEL="mistralai/mistral-7b-instruct" # 8,192
MODEL = "huggingfaceh4/zephyr-7b-beta"  # 4,096
# MODEL="openchat/openchat-7b" # 8,192
# MODEL="gryphe/mythomist-7b" # 32,768

# ----- less than $0.5 -----

# $0.1425/$0.1425 per 1M tokens
# MODEL="open-orca/mistral-7b-openorca" # 8,192

# $0.1474/$0.1474 per 1M tokens
# MODEL="meta-llama/llama-2-13b-chat" # 4,096

# $0.15/$0.15 per 1M tokens
# MODEL="nousresearch/nous-hermes-llama2-13b" # 4,096

# $0.1555/$0.4666 per 1M tokens
# MODEL="mistralai/mistral-tiny" # 32,000

# $0.18/$0.18 per 1M tokens
# MODEL="teknium/openhermes-2-mistral-7b" # 4,096
# MODEL="teknium/openhermes-2.5-mistral-7b" # 4,096

# $0.225/$0.225 per 1M tokens
# MODEL="gryphe/mythomax-l2-13b" # 4,096

# ----- more than $0.5 -----

# $0.27/$0.27 per 1M tokens
# MODEL="mistralai/mixtral-8x7b-instruct" #⭐️ 32,768
# MODEL="cognitivecomputations/dolphin-mixtral-8x7b" #️️⭐️ 32,000
# MODEL="undi95/remm-slerp-l2-13b" # 4,096

# $0.3/$0.3 per 1M tokens
# MODEL="nousresearch/nous-hermes-2-mixtral-8x7b-dpo" #⭐️ 32,768
# MODEL="nousresearch/nous-hermes-2-mixtral-8x7b-sft" #⭐️ 32,768

# $0.25/$0.5 per 1M tokens
# MODEL="google/gemini-pro" #⭐️ 131,040
# MODEL="google/gemini-pro-vision" #⭐️ 65,536

# $0.4/$0.4 per 1M tokens
# MODEL="meta-llama/codellama-34b-instruct" # 8,192
# MODEL="phind/phind-codellama-34b" #⭐️ 4,096

# ----- more than $1.0 -----

# $0.54/$0.54 per 1M tokens
# MODEL="mistralai/mixtral-8x7b" # 32,768

# $0.72/$0.72 per 1M tokens
# MODEL="01-ai/yi-34b-chat" #⭐️ 4,096
# MODEL="nousresearch/nous-hermes-yi-34b" #⭐️ 4,096

# $0.7/$0.9 per 1M tokens
# MODEL="meta-llama/llama-2-70b-chat" #⭐️ 4,096
# MODEL="jondurbin/airoboros-l2-70b" #⭐️ 4,096

# $0.81/$0.81 per 1M tokens
# MODEL="nousresearch/nous-hermes-llama2-70b" #⭐️ 4,096

# ----- more than $2.0 -----

# $1.125/$1.125 per 1M tokens
# MODEL="pygmalionai/mythalion-13b" # 8,192
# MODEL="gryphe/mythomax-l2-13b-8k" # 8,192
# MODEL="undi95/remm-slerp-l2-13b-6k" # 6,144

# $0.6666/$2 per 1M tokens
# MODEL="mistralai/mistral-small" 32,000

# ----- more than $3.0 -----

# $1/$2 per 1M tokens
# MODEL="openai/gpt-3.5-turbo" # 4,095
# MODEL="openai/gpt-3.5-turbo-1106" #⭐️ 16,385

# $1.5/$2 per 1M tokens
# MODEL="openai/gpt-3.5-turbo-instruct" # 4,095

# $2.25/$2.25 per 1M tokens
# MODEL="neversleep/noromaid-20b" # 8,192

# $3/$3 per 1M tokens
# MODEL="jondurbin/bagel-34b" #⭐️ 8,000
# MODEL="neversleep/noromaid-mixtral-8x7b-instruct" # 8,000

# $1.6/$5 per 1M tokens
# MODEL="anthropic/claude-instant-v1" #⭐️ 100,000

# $3.75/$3.75 per 1M tokens
# MODEL="migtissera/synthia-70b" # 8,192

# ----- more than $10.0 -----

# $5/$5 per 1M tokens
# MODEL="intel/neural-chat-7b" # 4,096
# MODEL="haotian-liu/llava-13b" # 2,048
# MODEL="nousresearch/nous-hermes-2-vision-7b" # 4,096

# $2.778/$8.333 per 1M tokens
# MODEL="mistralai/mistral-medium" #⭐️ 100,000

# $9.375/$9.375 per 1M tokens
# MODEL="alpindale/goliath-120b" # 6,144

# $8/$24 per 1M tokens
# MODEL="anthropic/claude-2" # 200,000

# $30/$60 per 1M tokens
# MODEL="openai/gpt-4" # 8,191

# $60/$120 per 1M tokens
# MODEL="openai/gpt-4-32k" # 32,767


def call_openrouter(args):

    client = OpenAI(
        base_url=args.api_url or "https://openrouter.ai/api/v1",
        api_key=args.api_key or os.getenv("OPENROUTER_API_KEY"),
    )

    stream = args.stream
    prompt = args.prompt or "Hello, how are you?"
    extra_headers = {
        "HTTP-Referer": args.site_url,  #$YOUR_SITE_URL, # Optional, for including your app on openrouter.ai rankings.
        "X-Title": args.app_name,  #$YOUR_APP_NAME, # Optional. Shows in rankings on openrouter.ai.
    }

    messages = [
        {
            "role": "user",
            "content": prompt,
        },
    ]

    completion_kwargs = dict(
        extra_headers=extra_headers,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        stream=stream,
        messages=messages,
    )

    completion = client.chat.completions.create(**completion_kwargs)

    if stream:
        full_text = ""
        for chunk in completion:
            if chunk.choices[0].delta:
                chunk_content = chunk.choices[0].delta.content
                sys.stdout.write(chunk_content)
                sys.stdout.flush()
    else:
        print(completion.choices[0].message.content)


def get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument("--model", type=str, default=MODEL, help="Model name")
    parser.add_argument("--api_url", type=str, help="OpenRouter API URL")
    parser.add_argument("--api_key", type=str, help="OpenRouter API Key")
    parser.add_argument("--stream", action="store_true", help="Stream or not")
    parser.add_argument("--site_url", type=str, default="https://github.com/uukuguy/speechless", help="Your site URL")
    parser.add_argument("--app_name", type=str, default="Speechless.AI", help="Your app name")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Max tokens")
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature")
    parser.add_argument("--prompt", type=str, help="Prompt")

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    print(args)
    call_openrouter(args)


if __name__ == "__main__":
    main()
