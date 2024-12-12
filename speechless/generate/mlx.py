#!/usr/bin/env python
"""
python -m speechless.generate.mlx --model_path dolphin-2.6-mistral-7b-dpo-laser-4bit-mlx  --verbose --max_new_tokens 512 --prompt_file $SPEECHLESS_ROOT/speechless/generate/prompts/hello_llm_en.txt
"""
import os, json
import mlx_lm
import mlx.core as mx

def colorprint(color, s):
    color_codes = {
        "black": 30,
        "red": 31,
        "green": 32,
        "yellow": 33,
        "blue": 34,
        "magenta": 35,
        "cyan": 36,
        "white": 39,
    }
    ccode = color_codes.get(color, 30)
    print(f"\033[1m\033[{ccode}m{s}\033[0m", end="", flush=True)


def colorprint_by_t0(s, t0):
    if t0 > 0.95:
        color = "white"
    elif t0 > 0.70:
        color = "green"
    elif t0 > 0.30:
        color = "yellow"
    else:
        color = "red"
    colorprint(color, s)


def generate(args):
    mx.random.seed(args.seed)

    # Building tokenizer_config
    tokenizer_config = {"trust_remote_code": True if args.trust_remote_code else None}
    if args.eos_token is not None:
        tokenizer_config["eos_token"] = args.eos_token

    model, tokenizer = mlx_lm.load(args.model_path, adapter_path=args.adapter_path, tokenizer_config=tokenizer_config)

    if args.prompt_file:
        prompt = open(args.prompt_file).read().strip()
    else:
        prompt = args.prompt

    if not args.ignore_chat_template and (
        hasattr(tokenizer, "apply_chat_template")
        and tokenizer.chat_template is not None
    ):
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    formatter = colorprint_by_t0 if args.verbose else None

    gen_kwargs = {
        'temp': args.temperature,
        'max_tokens': args.max_new_tokens,
    }

    response = mlx_lm.generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        verbose=args.verbose,
        formatter=formatter,
        **gen_kwargs,
    )
    print(response)

class MLX_API:
    def __init__(self, model, adapter_path=None, gen_params=None, lazy=True, eos_token=None, trust_remote_code=True, ignore_chat_template=False, verbose=False, seed=42):
        self.model_path = model
        self.adapter_path = adapter_path
        self.eos_token = eos_token
        self.trust_remote_code = trust_remote_code
        self.seed = seed
        self.verbose = verbose

        mx.random.seed(self.seed)
        # Building tokenizer_config
        self.tokenizer_config = {"trust_remote_code": True if self.trust_remote_code else None}
        if self.eos_token is not None:
            self.tokenizer_config["eos_token"] = self.eos_token

        self.model, self.tokenizer = mlx_lm.load(self.model_path, 
                                                 adapter_path=self.adapter_path,
                                                 tokenizer_config=self.tokenizer_config,
                                                 lazy=lazy)

        self.gen_arams = gen_params


    def generate(self, prompt, system_message="", gen_params=None, stream=False, ignore_chat_template=False, verbose=False):
        if not ignore_chat_template and (
            hasattr(self.tokenizer, "apply_chat_template")
            and self.tokenizer.chat_template is not None
        ):
            messages = [{"role": "system", "content": system_message}] if system_message else []
            messages.append({"role": "user", "content": prompt})
                
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        if gen_params is None:
            gen_params = self.gen_arams

        if gen_params is None:
            gen_params={
                "temperature": 0.7,
                # "top_p": 0.9,
                "min_p": 0.1,
                "max_tokens": 512,
            }

        mlx_gen_params = {
            'temp': gen_params.get("temperature", 0.7),
            'max_tokens': gen_params.get("max_tokens", 1024),
            "top_p": gen_params.get("top_p", 0.9),
            "min_p": gen_params.get("min_p", 0.1),
        }
    
        if stream:
            stream_generator = mlx_lm.stream_generate(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=prompt,
                **mlx_gen_params,
            )
            for chunk in stream_generator:
                yield chunk
        else:
            response = mlx_lm.generate(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=prompt,
                verbose=verbose,
                **mlx_gen_params,
            )

            generated_text = response
            return generated_text


def get_args():
    from argparse import ArgumentParser

    parser = ArgumentParser(description="MLX: Machine Learning eXperiment")
    parser.add_argument("--trust-remote-code", action="store_true", help="Enable trusting remote code for tokenizer")
    parser.add_argument("--eos-token", type=str, default=None, help="End of sequence token for tokenizer")
    parser.add_argument("--ignore-chat-template", action="store_true", help="Use the raw prompt without the tokenizer's chat template.")
    # parser.add_argument("--colorize", action="store_true", help="Colorize output based on T[0] probability")

    parser.add_argument("--model_path", type=str, required=True, help="HF model path")
    parser.add_argument("--adapter_path", type=str, help="adapter file path")
    parser.add_argument("--prompt", type=str, help="prompt to run")
    parser.add_argument("--prompt_file", type=str, help="prompt file")

    parser.add_argument("--temperature", type=float, default=0.7, help="temperature")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="max new tokens")
    parser.add_argument("--ctx_size", type=int, default=16384, help="context size")

    parser.add_argument("--top_p", type=float, default=0.9, help="top p")
    parser.add_argument("--top_k", type=int, default=40, help="top k")
    parser.add_argument("--repeat_penalty", type=float, default=1.1, help="repeat penalty")

    parser.add_argument("--verbose", action="store_true", help="verbose")
    parser.add_argument("--seed", type=int, default=0, help="seed")

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    generate(args)


if __name__ == '__main__':
    main()
