"""
{{- if or .System .Tools }}<|im_start|>system
{{- if .System }}
{{ .System }}
{{- end }}
{{- if .Tools }}

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{{- range .Tools }}
{"type": "function", "function": {{ .Function }}}
{{- end }}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
{{- end }}<|im_end|>
{{ end }}
"""
"""
sample = format_chat(dataset[0]["messages"])
print("Formatted sample:\n", sample)
print("Tokenized sample:\n", tokenizer.decode(tokenizer(sample)["input_ids"]))
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_temperature",
            "description": "Get current temperature at a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": 'The location to get the temperature for, in the format "City, State, Country".',
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": 'The unit to return the temperature in. Defaults to "celsius".',
                    },
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_temperature_date",
            "description": "Get temperature at a location and date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": 'The location to get the temperature for, in the format "City, State, Country".',
                    },
                    "date": {
                        "type": "string",
                        "description": 'The date to get the temperature for, in the format "Year-Month-Day".',
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": 'The unit to return the temperature in. Defaults to "celsius".',
                    },
                },
                "required": ["location", "date"],
            },
        },
    },
]
# MESSAGES = [
#     {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\nCurrent Date: 2024-09-30"},
#     {"role": "user",  "content": "What's the temperature in San Francisco now? How about tomorrow?"}
# ]
# MESSAGES = [
#     {'role': 'system', 'content': 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\nCurrent Date: 2024-09-30'},
#     {'role': 'user', 'content': "What's the temperature in San Francisco now? How about tomorrow?"},
#     {'role': 'assistant', 'content': '', 'tool_calls': [
#         {'type': 'function', 'function': {'name': 'get_current_temperature', 'arguments': {'location': 'San Francisco, CA, USA'}}}, 
#         {'type': 'function', 'function': {'name': 'get_temperature_date', 'arguments': {'location': 'San Francisco, CA, USA', 'date': '2024-10-01'}}},
#     ]},
#     {'role': 'tool', 'name': 'get_current_temperature', 'content': '{"temperature": 26.1, "location": "San Francisco, CA, USA", "unit": "celsius"}'},
#     {'role': 'tool', 'name': 'get_temperature_date', 'content': '{"temperature": 25.9, "location": "San Francisco, CA, USA", "date": "2024-10-01", "unit": "celsius"}'},
# ]
MESSAGES1 = [
    {'role': 'system', 'content': 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\nCurrent Date: 2024-09-30', 'tools': TOOLS},
    {'role': 'user', 'content': "What's the temperature in San Francisco now? How about tomorrow?"},
    {'role': 'assistant', 'content': '', 'tool_calls': [
        {'type': 'function', 'function': {'name': 'get_current_temperature', 'arguments': {'location': 'San Francisco, CA, USA'}}}, 
    ]},
    {'role': 'tool', 'name': 'get_current_temperature', 'content': '{"temperature": 26.1, "location": "San Francisco, CA, USA", "unit": "celsius"}'},
]
MESSAGES2 = [
    {'role': 'system', 'content': 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\nCurrent Date: 2024-09-30', 'tools': TOOLS},
    {'role': 'user', 'content': "What's the temperature in San Francisco now? How about tomorrow?"},
    {'role': 'assistant', 'content': '', 'tool_calls': [
        {'type': 'function', 'function': {'name': 'get_temperature_date', 'arguments': {'location': 'San Francisco, CA, USA', 'date': '2024-10-01'}}},
    ]},
    {'role': 'tool', 'name': 'get_temperature_date', 'content': '{"temperature": 25.9, "location": "San Francisco, CA, USA", "date": "2024-10-01", "unit": "celsius"}'},
]


import json
from typing import Dict, Sequence
import torch
from torch.nn.utils.rnn import pad_sequence
import transformers
from dataclasses import dataclass


def format_chat(messages, output_format, add_generation_prompt=False, include_toolcall_example=True):
    """
    output_format: "json" or "toml"
    """
    formatted = []
    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls", [])

        if role == "system":
            system_parts = []
            if content:
                system_parts.append(content)
            if "tools" in msg:
                system_parts.append("\n# Tools\nYou may call one or more functions...")
                system_parts.append("<tools>")
                for tool in msg["tools"]:
                    system_parts.append(json.dumps(tool, ensure_ascii=False))

                system_parts.append("</tools>")

                if include_toolcall_example:
                    if output_format == "json":
                        system_parts.append("")
                        system_parts.append("For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:")
                        system_parts.append("<tool_call>")
                        system_parts.append("{\"name\": <function-name>, \"arguments\": <args-json-object>}")
                        system_parts.append("</tool_call>")
                        system_parts.append("")
                    elif output_format == "toml":
                        system_parts.append("")
                        system_parts.append("For each function call, return a toml object with function name and arguments within <tool_call></tool_call> XML tags:")
                        system_parts.append("<tool_call>")
                        system_parts.append("[[functions]]")

                        system_parts.append("name = <function-name>")
                        system_parts.append("")
                        system_parts.append("[functions.arguments]")
                        system_parts.append("\"<arg-name>\" = <arg-value>")

                        # system_parts.append("name = \"send_mail\"")
                        # # system_parts.append("name = <function-name>")
                        # system_parts.append("")
                        # system_parts.append("[functions.arguments]")
                        # # system_parts.append("\"<arg-name>\" = <arg-value>")
                        # system_parts.append("address = \"example@gmail.com\"")
                        # system_parts.append("subject = \"Meeting\"")
                        # # List example
                        # system_parts.append("peoples = [\"Tom\", \"Jerry\"]")
                        # # Dict example
                        # system_parts.append("objects = {\"machine\" = 1000, \"labor\" = 500}")
                        # # Date example
                        # system_parts.append("start_date = \"2024-10-01\"")
                        # # Time example
                        # system_parts.append("end_time = \"2025-11-01 08:00:00\"")

                        system_parts.append("</tool_call>")
                        system_parts.append("")
                    else:
                        raise ValueError(f"Unknown output format in format_chat(): {output_format}")

            formatted.append(f"<|im_start|>system\n{'\n'.join(system_parts)}<|im_end|>\n")

        elif role == "user":
            formatted.append(f"<|im_start|>user\n{content}<|im_end|>")

        elif role == "assistant":
            assistant_lines = []
            if tool_calls:
                assistant_lines.append("<tool_call>")
                for call in tool_calls:
                    assistant_lines.append(
                        json.dumps({
                            "name": call["function"]["name"],
                            "arguments": call["function"]["arguments"]
                        },
                                   ensure_ascii=False)
                    )
                assistant_lines.append("</tool_call>")
            if content:
                assistant_lines.append(content)
            formatted.append(f"<|im_start|>assistant\n{'\n'.join(assistant_lines)}<|im_end|>")

        elif role == "tool":
            formatted.append(f"<|im_start|>user\n<tool_response>\n{content}\n</tool_response><|im_end|>")

    result = "\n".join(formatted)
    if add_generation_prompt:
        result += "<|im_start|>assistant\n"
    # print(f"{result}")
    return result


@dataclass
class QwenMultiRoundsDataCollator(object):
    tokenizer: transformers.PreTrainedTokenizer
    model_max_length: int
    prompt_type: str = None

    def preprocess(self, messages):
        formatted_text = format_chat(messages)
        tokenized = self.tokenizer(
            formatted_text, truncation=True, max_length=self.model_max_length, add_special_tokens=False
        )

        # 生成标签（仅计算assistant部分的损失）
        labels = []
        current_role = None
        for token_id in tokenized["input_ids"]:
            token = self.tokenizer.decode([token_id])
            # print(f"{token=},{current_role=}")
            # print(token, flush=True)
            if token == "<|im_start|>":
                labels.append(-100)
                current_role = None
            elif current_role is None:
                current_role = token.strip()
                labels.append(-100)
            elif token == "<|im_end|>":
                labels.append(-100)
                current_role = None
            else:
                if current_role == "assistant":
                    labels.append(token_id)
                else:
                    labels.append(-100)

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels
        }

    def __call__(self, examples: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = []
        labels = []
        batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": []
        }

        new_examples = []
        for ex in examples:
            messages = ex['messages']
            if isinstance(messages, str):
                messages = json.loads(messages)
            new_examples.append(self.preprocess(messages))
        examples = new_examples
        # examples = [ self.preprocess(ex["messages"]) for ex in examples]
        max_len = min(max(len(ex["input_ids"]) for ex in examples), self.model_max_length)
        # max_len = self.model_max_length

        # for ex in examples:
        #     pad_len = max_len - len(ex["input_ids"])
        #     input_ids = ex["input_ids"] + [self.tokenizer.pad_token_id] * pad_len
        #     batch["input_ids"].append(torch.tensor(input_ids))

        #     # 填充attention_mask
        #     attention_mask = ex["attention_mask"] + [0] * pad_len
        #     batch["attention_mask"].append(torch.tensor(attention_mask))

        #     # 填充labels
        #     labels = ex["labels"] + [-100] * pad_len
        #     batch["labels"].append(torch.tensor(labels))

        # Apply padding
        input_ids = [torch.tensor(ex["input_ids"][:max_len]) for ex in examples]  
        labels = [torch.tensor(ex["labels"][:max_len]) for ex in examples]
        if self.tokenizer.padding_side == "left":
            input_ids = [t.flip(-1) for t in input_ids]
            labels = [t.flip(-1) for t in labels]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        if self.tokenizer.padding_side == "left":
            input_ids = input_ids.flip(-1)
            labels = labels.flip(-1)

        

        # return {
        #     "input_ids": torch.stack(batch["input_ids"]),
        #     # "attention_mask": torch.stack(batch["attention_mask"]),
        #     'attention_mask': batch["input_ids"].ne(self.tokenizer.pad_token_id),
        #     "labels": torch.stack(batch["labels"])
        # }
        return {
            "input_ids": input_ids,
            'attention_mask': input_ids.ne(self.tokenizer.pad_token_id),
            "labels": labels
        }


def test(args):
    from datasets import load_dataset, Dataset
    from speechless.finetune.dataset_utils.dataset_utils import load_tokenizer
    tokenizer = load_tokenizer(args.model_name)

    special_tokens = ["<|im_start|>", "<|im_end|>", "<tool_call>", "<tool_response>", "assistant"]
    # tokenizer.add_tokens(special_tokens)
    if not all(token in tokenizer.all_special_tokens for token in special_tokens):
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        # model.resize_token_embeddings(len(tokenizer)) # 适配新增token

    # dataset = load_dataset("jsonl", data_files=args.data_file)
    dataset = Dataset.from_dict(
        {"train": [
            # {"messages": MESSAGES1},
            # {"messages": MESSAGES2},
            {"messages": json.dumps(MESSAGES1, ensure_ascii=False)},
            {"messages": json.dumps(MESSAGES2, ensure_ascii=False)},
            ]}
        )

    data_collator = QwenMultiRoundsDataCollator(
        tokenizer=tokenizer,
        model_max_length=1024,
        prompt_type="qwen2.5",
    )
    examples = dataset["train"]
    data = data_collator(examples)
    print(f"{examples=}")
    print(f"{data=}")


def get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--data_file", type=str, default=None)
    parser.add_argument(
        # "--model_name", type=str, default="/opt/local/llm_models/huggingface.co/google-bert/bert-base-chinese"
        "--model_name", type=str, default="/opt/local/llm_models/huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    test(args)
