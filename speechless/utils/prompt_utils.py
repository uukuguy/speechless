#!/usr/bin/env python
"""
Provide the generate_llm_prompt function to facilitate generating final prompts for LLM based on different standard prompt templates.

Dependency:

$ pip install fastchat

Usage: 

$ python -m prompt_utils --prompt_template alpaca --instruction "What is the capital of France?" 

```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction: What is the capital of France?

### Response:
```

$ python -m prompt_utils --prompt_template general-llama-2 --instruction "What is the capital of France?" 

```
[INST] <<SYS>>
You are a helpful assistant.
<</SYS>>

What is the capital of France? [/INST]
```

$ python -m prompt_utils --prompt_template general-chatml --instruction "What is the capital of France?" 

```
<|im_start|>system
You are a helpful, respectful, and honest assistant. Write out your reasoning step-by-step to be sure you get the right answers!<|im_end|>
<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant
```

Useful prompt templates:

    # general-llama-2, general-mistral, general-chatml
    # vicuna_v1.1, alpaca, chatglm3, deepseek-coder, stablelm
    # chatgpt, claude, metamath, bard, gemini
    # falcon, internlm-chat, starchat, baichuan2-chat, mistral, llama-2, open-orca, mistral-7b-openorca
    # dolphin-2.2.1-mistral-7b, OpenHermes-2.5-Mistral-7B, Nous-Hermes-2-Mixtral-8x7B-DPO
    # qwen-7b-chat, Yi-34b-chat, phind, zephyr, TinyLlama, orca-2, deepseek-chat, solar, llava-chatml, gemma

"""
from fastchat.conversation import Conversation, SeparatorStyle, register_conv_template, get_conv_template, conv_templates

# General llama2 template
# reference: https://huggingface.co/blog/codellama#conversational-instructions
# reference: https://github.com/facebookresearch/llama/blob/1a240688810f8036049e8da36b073f63d2ac552c/llama/generation.py#L212
register_conv_template(
    Conversation(
        name="general-llama-2",
        system_template="[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n",
        system_message="You are a helpful assistant.",
        roles=("[INST]", "[/INST]"),
        sep_style=SeparatorStyle.LLAMA2,
        sep=" ",
        sep2=" </s><s>",
    )
)

# General Mistral template
# source: https://docs.mistral.ai/llm/mistral-instruct-v0.1#chat-template
register_conv_template(
    Conversation(
        name="general-mistral",
        system_template="[INST] {system_message}\n",
        roles=("[INST]", "[/INST]"),
        sep_style=SeparatorStyle.LLAMA2,
        sep=" ",
        sep2="</s>",
    )
)

# General ChatML template
# Open-Orca/Mistral-7B-OpenOrca template
# source: https://huggingface.co/Open-Orca/Mistral-7B-OpenOrca
# reference: https://huggingface.co/Open-Orca/Mistral-7B-OpenOrca#prompt-template
register_conv_template(
    Conversation(
        name="general-chatml",
        system_template="<|im_start|>system\n{system_message}",
        system_message=
        "You are a helpful, respectful, and honest assistant. Write out your reasoning step-by-step to be sure you get the right answers!",
        roles=("<|im_start|>user", "<|im_start|>assistant"),
        sep_style=SeparatorStyle.CHATML,
        sep="<|im_end|>",
        stop_token_ids=[32000, 32001],
    )
)

available_templates = sorted(list(conv_templates.keys()))


def generate_llm_prompt(prompt_template, instruction, system_message=None, history=None):
    conv = get_conv_template(prompt_template)
    if system_message:
        conv.set_system_message(system_message)

    if history:
        for (instruction, response) in history:
            conv.append_message(conv.roles[0], instruction)
            conv.append_message(conv.roles[1], response)

    conv.append_message(conv.roles[0], instruction)
    conv.append_message(conv.roles[1], None)

    prompt = conv.get_prompt()

    return prompt


def get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--prompt_template", type=str, choices=available_templates, help="Prompt template")
    parser.add_argument("--instruction", required=True, type=str, help="Instruction")
    parser.add_argument("--system_message", type=str, help="System message")
    return parser.parse_args()


def main():
    args = get_args()
    # print(f"Available prompt templates: {available_templates}")
    prompt = generate_llm_prompt(args.prompt_template, args.instruction, system_message=args.system_message)
    print(prompt)


if __name__ == "__main__":
    main()
