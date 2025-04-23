from typing import Dict, Sequence
import copy
import torch
import transformers
from dataclasses import dataclass
from torch.nn.utils.rnn import pad_sequence
from loguru import logger

IGNORE_INDEX = -100


def preprocess_toolbench_dataset(
    example,
    model_max_length: int,
    tokenizer: transformers.PreTrainedTokenizer,
    template: str = "tool-llama-single-round"
) -> Dict:
    # conv = get_conversation_template(template)
    # if template == "tool-llama":
    #     roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    # elif template == "tool-llama-single-round" or template == "tool-llama-multi-rounds":
    #     roles = {"system": conv.roles[0], "user": conv.roles[1], "function": conv.roles[2], "assistant": conv.roles[3]}

    roles = {
        "system": "System",
        "user": "User",
        "function": "Function",
        "assistant": "Assistant"
    }
    seps = ["\n", "</s>"]

    # Apply prompt templates
    conversation = ""
    for i, round in enumerate(example['conversations']):
        role = roles[round['from']]
        message = round['value']
        if i + 1 == len(example['conversations']) and message:
            conversation += role + ": " + str(message) + seps[1]
        elif message:
            conversation += role + ": " + str(message) + seps[0]
        else:
            conversation += role + ":"

    # Tokenize conversations
    input_ids = tokenizer(
        conversation,
        # return_tensors="pt",
        # padding="max_length",
        max_length=model_max_length,
        truncation=True,
        # add_special_tokens=False,
    ).input_ids
    # input_ids begin with '<s>' and end with '</s>'
    assert input_ids[0] == tokenizer.bos_token_id and input_ids[-1] == tokenizer.eos_token_id
    labels = copy.deepcopy(input_ids)

    input_ids = torch.tensor(input_ids)
    labels = torch.tensor(labels)

    # Mask targets. Only compute loss on the assistant outputs.
    sep = seps[0] + roles['assistant'] + ": "

    total_len = int(labels.ne(tokenizer.pad_token_id).sum())
    turns = conversation.split(seps[1])
    cur_len = 1
    labels[:cur_len] = IGNORE_INDEX
    for i, turn in enumerate(turns):
        if turn == "":
            continue
        turn_len = len(tokenizer(turn).input_ids)

        parts = turn.split(sep)

        # only train on the last assistant reply, treat the history chat as instruction
        prefix = parts[:-1]
        instruction = ""
        for part in prefix:
            instruction += part
            instruction += sep

        # "-2" is hardcoded for the LLaMA tokenizer to make the offset correct.
        instruction_len = len(tokenizer(instruction).input_ids) - 2

        # Ignore the user instructions
        labels[cur_len:cur_len + instruction_len] = IGNORE_INDEX
        cur_len += turn_len

    labels[cur_len:] = IGNORE_INDEX

    # if False:  # Inspect and check the correctness of masking
    #     z = labels.clone()
    #     z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
    #     rank0_print(tokenizer.decode(z))

    if cur_len < model_max_length:
        if cur_len != total_len:
            labels[:] = IGNORE_INDEX
            logger.warning(
                f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                f" (ignored)"
                f"{conversation=}"
            )
    return dict(
        input_ids=input_ids,
        labels=labels,
        # attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


def preprocess_multi_rounds_dialog(
    example,
    model_max_length: int,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # conv = get_conversation_template(template)
    # if template == "tool-llama":
    #     roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    # elif template == "tool-llama-single-round" or template == "tool-llama-multi-rounds":
    #     roles = {"system": conv.roles[0], "user": conv.roles[1], "function": conv.roles[2], "assistant": conv.roles[3]}

    roles = {
        "system": "System",
        "user": "User",
        "function": "Function",
        "assistant": "Assistant"
    }
    seps = ["\n", "</s>"]

    # The dialogue process is divided into multiple rounds, with each round ending when the Assistant speaks.
    dialog = example['conversations']
    dialog_rounds = []
    round_messages = []
    for i, round in enumerate(dialog):
        who = round['from']
        message = round['value']
        if who != 'assistant':
            round_messages.append((who, message))
        else:
            dialog_rounds.append({
                'round_messages': round_messages,
                'assistant': message,
            })
            round_messages = []
    if len(round_messages) > 0:
        logger.warning(f"WARNING: the last round is not ended by the assistant. IGNORE!!!. {dialog=}")
        dialog_rounds = []
    # print(f"{dialog_rounds=}")

    example_input_ids = None
    example_output_ids = None
    for idx, round in enumerate(dialog_rounds):
        round_messages = round['round_messages']
        assistant_message = round['assistant']
        source = ""
        for (who, message) in round_messages:
            source += roles[who] + ": " + str(message) + seps[0]
        source += roles['assistant'] + ": "
        target = assistant_message + tokenizer.eos_token

        # source = f"{tokenizer.bos_token}{source}"
        # target = f"{bot_response.strip()}\n{tokenizer.eos_token}"

        tokenized_source = tokenizer(source, max_length=model_max_length, truncation=True, add_special_tokens=False)
        tokenized_target = tokenizer(target, max_length=model_max_length, truncation=True, add_special_tokens=False)
        tokenized_input = torch.tensor(tokenized_source['input_ids'] + tokenized_target['input_ids'])
        tokenized_output = torch.tensor([IGNORE_INDEX for _ in range(len(tokenized_source['input_ids']))] +
                                        copy.deepcopy(tokenized_target['input_ids']))
        if idx == 0:
            example_input_ids = tokenized_input
            example_output_ids = tokenized_output
        else:
            example_input_ids = torch.concatenate((example_input_ids, tokenized_input), dim=0)
            example_output_ids = torch.concatenate((example_output_ids, tokenized_output), dim=0)

    input_ids = example_input_ids
    labels = example_output_ids
    return dict(
        input_ids=input_ids,
        labels=labels,
        # attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


def generate_round_prompt_toolllama(
    idx: int,
    human_input: str,
    bot_response: str,
    bos_token: str,
    eos_token: str,
    system_prompt: str = None,
):
    if idx == 0:
        if system_prompt:
            source = f"{system_prompt} Human: {human_input} Assistant: "
        else:
            system_prompt = "A chat between a curious user and an artificial intelligence assistant who can use external tools and APIs to solve the user's question."
            "The assistant gives tools and APIs calling processes or final answer to the human's question."
            human_input = "Human: {instruction} Assistant: ".format(instruction=human_input)
            source = f"{system_prompt} {human_input}"
    else:
        human_input = "Human: {instruction} Assistant: ".format(instruction=human_input)
        source = f"{human_input}"
    source = f"{bos_token}{source}"
    target = f"{bot_response.strip()}\n{eos_token}"

    return source, target

def generate_round_prompt_raw(
    idx: int,
    human_input: str,
    bot_response: str,
    bos_token: str,
    eos_token: str,
    system_prompt: str = None,
):
    source = f"{bos_token}{human_input}"
    target = f"{bot_response.strip()}\n{eos_token}"

    return source, target

def generate_round_prompt_alpaca(
    idx: int,
    human_input: str,
    bot_response: str,
    bos_token: str,
    eos_token: str,
    system_prompt: str = None,
):
    if idx == 0:
        if system_prompt:
            source = f"{system_prompt}\n\n### Instruction:\n{human_input}\n\n### Response:"
        else:
            system_prompt = "Below is an instruction that describes a task.\nWrite a response that appropriately completes the request.\n\n"
            human_input = "### Instruction:\n{instruction}\n\n### Response:".format(instruction=human_input)
            source = f"{system_prompt}{human_input}"
    else:
        human_input = "### Instruction:\n{instruction}\n\n### Response:".format(instruction=human_input)
        source = f"{human_input}"

    source = f"{bos_token}{source}"
    target = f"{bot_response.strip()}\n{eos_token}"

    return source, target


def generate_round_prompt_llama2(
    idx: int,
    human_input: str,
    bot_response: str,
    bos_token: str,
    eos_token: str,
    system_prompt: str = None,
):
    if idx == 0:
        if system_prompt:
            source = f"{bos_token}[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n{human_input}[/INST]"
        else:
            source = f"{bos_token}[INST]{human_input}[/INST]"
    else:
        source = f"[INST]{human_input}[/INST]"

    target = f"{bot_response.strip()}\n{eos_token}"

    return source, target


def generate_round_prompt_chatlm(
    idx: int,
    human_input: str,
    bot_response: str,
    bos_token: str,
    eos_token: str,
    system_prompt: str = None,
):
    # f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant"
    if idx == 0:
        # if system_prompt:
        #     # source = f"{system_prompt}\n\n### Instruction:\n{human_input}\n\n### Response:"
        #     source = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        # else:
        #     # system_prompt = "Below is an instruction that describes a task.\nWrite a response that appropriately completes the request.\n\n"
        system_prompt = "You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior."
        human_input = f"<|im_start|>user\n{human_input}<|im_end|>\n<|im_start|>assistant"
        source = f"<|im_start|>system\n{system_prompt}<|im_end|>\n{human_input}"
    else:
        human_input = f"<|im_start|>user\n{human_input}<|im_end|>\n<|im_start|>assistant"
        source = f"{human_input}"

    source = f"{bos_token}{source}"
    target = f"{bot_response.strip()}\n{eos_token}"

    return source, target


def generate_round_prompt_minicpm(
    idx: int,
    human_input: str,
    bot_response: str,
    bos_token: str,
    eos_token: str,
    system_prompt: str = None,
):
    source = f"{bos_token}<用户>{human_input}<AI>"
    target = f"{bot_response.strip()}{eos_token}"

    return source, target


@dataclass
class DataCollatorForCausalLM(object):
    tokenizer: transformers.PreTrainedTokenizer
    model_max_length: int

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract elements
        sources = [f"{self.tokenizer.bos_token}{example['input']}" for example in instances]
        targets = [f"{example['output']}{self.tokenizer.eos_token}" for example in instances]

        # Tokenize
        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=self.model_max_length,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_targets = self.tokenizer(
            targets,
            max_length=self.model_max_length,
            truncation=True,
            add_special_tokens=False,
        )

        # Build the input and labels for causal LM
        input_ids = []
        labels = []
        for tokenized_source, tokenized_target in zip(
            tokenized_sources_with_prompt['input_ids'], tokenized_targets['input_ids']
        ):
            input_ids.append(torch.tensor(tokenized_source + tokenized_target))
            labels.append(
                torch.tensor([IGNORE_INDEX for _ in range(len(tokenized_source))] + copy.deepcopy(tokenized_target))
            )
        # Apply padding
        if self.tokenizer.padding_side == "left":
            input_ids = [t.flip(-1) for t in input_ids]
            labels = [t.flip(-1) for t in labels]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        if self.tokenizer.padding_side == "left":
            input_ids = input_ids.flip(-1)
            labels = labels.flip(-1)

        data_dict = {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': input_ids.ne(self.tokenizer.pad_token_id),
        }
        return data_dict


@dataclass
class DialogDataCollatorForCausalLM(object):
    tokenizer: transformers.PreTrainedTokenizer
    model_max_length: int
    prompt_type: str = None

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract elements
        input_ids = []
        labels = []
        for example in instances:
            # system_prompt = example.get('system_prompt', 'A Bot').strip() + "\n\n"
            if self.prompt_type is not None:
                prompt_type = self.prompt_type
            else:
                prompt_type = example.get('prompt_type', None)
            if prompt_type == "tool-llama-single-round":
                data_dict = preprocess_toolbench_dataset(
                    example,
                    model_max_length=self.model_max_length,
                    tokenizer=self.tokenizer,
                    template="tool-llama-single-round"
                )
                example_input_ids = data_dict['input_ids']
                example_labels = data_dict['labels']
                if example_input_ids is not None:
                    # print(f"{example_input_ids.shape=},{example_input_ids=}")
                    # print(f"{example_labels.shape=},{example_labels=}")
                    input_ids.append(example_input_ids)
                    labels.append(example_labels)
                continue

            elif prompt_type == "tool-llama-multi-rounds":
                data_dict = preprocess_multi_rounds_dialog(
                    example,
                    model_max_length=self.model_max_length,
                    tokenizer=self.tokenizer,
                )
                example_input_ids = data_dict['input_ids']
                example_labels = data_dict['labels']
                if example_input_ids is not None:
                    # print(f"{example_input_ids.shape=},{example_input_ids=}")
                    # print(f"{example_labels.shape=},{example_labels=}")
                    input_ids.append(example_input_ids)
                    labels.append(example_labels)
                continue

            system_prompt = example.get('system_prompt', "").strip()
            if system_prompt:
                system_prompt += "\n\n"
            example_input_ids = None
            example_output_ids = None

            if prompt_type == "llama3":
                pass
            else:
                # human_bot_dialog = example['dialog']
                human_bot_dialog = []
                dialog = example['conversations']
                for _i in range(len(dialog) // 2):
                    human_input = dialog[2 * _i]['value']
                    bot_output = dialog[2 * _i + 1]['value']
                    human_bot_dialog.append((human_input, bot_output))
                if len(human_bot_dialog) < 1:
                    continue
                for idx, round in enumerate(human_bot_dialog):
                    human_input, bot_response = round
                    if prompt_type == 'toolllama':
                        source, target = generate_round_prompt_toolllama(
                            idx,
                            human_input,
                            bot_response,
                            bos_token=self.tokenizer.bos_token,
                            eos_token=self.tokenizer.eos_token,
                            system_prompt=system_prompt,
                        )
                    elif prompt_type == "chatlm":
                        source, target = generate_round_prompt_chatlm(
                            idx,
                            human_input,
                            bot_response,
                            bos_token=self.tokenizer.bos_token,
                            eos_token=self.tokenizer.eos_token,
                            system_prompt=system_prompt,
                        )
                    elif prompt_type == "llama2":
                        source, target = generate_round_prompt_llama2(
                            idx,
                            human_input,
                            bot_response,
                            bos_token=self.tokenizer.bos_token,
                            eos_token=self.tokenizer.eos_token,
                            system_prompt=system_prompt,
                        )
                    elif prompt_type == "minicpm":
                        source, target = generate_round_prompt_minicpm(
                            idx,
                            human_input,
                            bot_response,
                            bos_token=self.tokenizer.bos_token,
                            eos_token=self.tokenizer.eos_token,
                            system_prompt=system_prompt,
                        )
                    elif prompt_type == "raw":
                        source, target = generate_round_prompt_raw(
                            idx,
                            human_input,
                            bot_response,
                            bos_token=self.tokenizer.bos_token,
                            eos_token=self.tokenizer.eos_token,
                            system_prompt=system_prompt,
                        )
                    elif prompt_type == "chat_template":
                        source = self.tokenizer.apply_chat_template(human_input, add_generation_prompt=True, tokenize=False)
                        target = bot_response
                    else:  # default alpaca
                        source, target = generate_round_prompt_alpaca(
                            idx,
                            human_input,
                            bot_response,
                            bos_token=self.tokenizer.bos_token,
                            eos_token=self.tokenizer.eos_token,
                            system_prompt=system_prompt,
                        )
                    

                    tokenized_source = self.tokenizer(
                        source,
                        max_length=self.model_max_length,
                        truncation=True,
                        add_special_tokens=False,
                    )
                    tokenized_target = self.tokenizer(
                        target,
                        max_length=self.model_max_length,
                        truncation=True,
                        add_special_tokens=False,
                    )
                    tokenized_input = torch.tensor(tokenized_source['input_ids'] + tokenized_target['input_ids'])
                    tokenized_output = torch.tensor([IGNORE_INDEX for _ in range(len(tokenized_source['input_ids']))] +
                                                    copy.deepcopy(tokenized_target['input_ids']))

                    # print(f"{source=}")
                    # print(f"{tokenized_input=}")
                    # print(f"{target=}")
                    # print(f"{tokenized_target=}")
                    if idx == 0:
                        example_input_ids = tokenized_input
                        example_output_ids = tokenized_output
                    else:
                        example_input_ids = torch.concatenate((example_input_ids, tokenized_input), dim=0)
                        example_output_ids = torch.concatenate((example_output_ids, tokenized_output), dim=0)

            input_ids.append(example_input_ids)
            labels.append(example_output_ids)

        # print(f"{example=}")
        # print(f"{input_ids=}")
        # print(f"{labels=}")
        # Apply padding
        if self.tokenizer.padding_side == "left":
            input_ids = [t.flip(-1) for t in input_ids]
            labels = [t.flip(-1) for t in labels]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        if self.tokenizer.padding_side == "left":
            input_ids = input_ids.flip(-1)
            labels = labels.flip(-1)

        data_dict = {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': input_ids.ne(self.tokenizer.pad_token_id),
        }
        return data_dict
