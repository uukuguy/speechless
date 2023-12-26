#!/usr/bin/env python
import os, json
import torch
from tqdm import tqdm
import gc, ctypes
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

def clean_memory():
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)
    torch.cuda.empty_cache()

# processed = 0

def main(args):

    if os.path.exists(args.output_file):
        raise FileExistsError(f"File {args.output_file} already exists.")

    print(f"Loading reward model from {args.reward_model_path} ...")
    rank_model, tokenizer = AutoModelForSequenceClassification.from_pretrained(args.reward_model_path).cuda(), AutoTokenizer.from_pretrained(args.reward_model_path)
    question, answer = "Explain nuclear fusion like I am five", "Nuclear fusion is the process by which two or more protons and neutrons combine to form a single nucleus. It is a very important process in the universe, as it is the source of energy for stars and galaxies. Nuclear fusion is also a key process in the production of energy for nuclear power plants."
    inputs = tokenizer(question, answer, return_tensors='pt').to("cuda")
    score = rank_model(**inputs).logits[0].detach()
    print(f"Test score: {float(score):.4f}")

    # print(f"Loading data from {args.input_file} ...")
    # dataset = load_dataset('json', data_files=args.input_file, split='train')
    # print(f"Total {len(dataset)} samples.")

    lines = open(args.input_file).readlines()
    num_lines = len(lines)
    print(f"Total {num_lines} samples.")

    with open(args.output_file, 'w') as fd:
        for idx, line in enumerate(tqdm(lines, ncols=100)):
            data = json.loads(line.strip())
            category = data['category']
            dialog = data['conversations']
            quality_score = 0.0
            try:
                if len(dialog) > 2:
                    for idx, round in enumerate(dialog):
                        who = round['from']
                        value = round['value']
                        # "### Instruction:\n{instruction}\n\n### Response: "
                        if idx < len(dialog) - 1:
                            if idx % 2 == 0:
                                question += f"### Instruction:\n{value}\n\n"
                            else:
                                question += f"### Response: {value}\n\n"
                        else:
                            question += f"### Response: "
                            answer = value
                else:
                    assert len(dialog) == 2
                    question = dialog[0]['value']
                    answer = dialog[1]['value']

                inputs = tokenizer(question, answer, return_tensors='pt').to("cuda")
                quality_score = float(rank_model(**inputs).logits[0].detach())

                if idx % 100 == 0:
                    clean_memory()

            except Exception as e:
                print(dialog)
                print(e)

            _out = {'category': category, 'conversations': dialog, 'quality_score': quality_score}
            # print(f"{_out=}")
            new_line = json.dumps(_out, ensure_ascii=False)
            fd.write(new_line + '\n')
    print(f"Saved {num_lines} samples to {args.output_file}.")

    print(f"Sorting data ...")
    sorted_file = args.input_file.replace(".jsonl", "") + '_sorted.jsonl'
    dataset = load_dataset('json', data_files=args.output_file, split='train')
    print(f"Total {len(dataset)} samples.")
    dataset = dataset.sort('quality_score', reverse=True)
    dataset.to_json(sorted_file, orient="records", lines=True, index=False)
    print(f"Saved {len(dataset)} samples to {sorted_file}.")

    # def _format(example):
    #     global processed
    #     quality_score = 0.0
    #     question = ""
    #     answer = ""
    #     dialog = example['conversations']
    #     try:
    #         if len(dialog) > 2:
    #             for idx, round in enumerate(dialog):
    #                 who = round['from']
    #                 value = round['value']
    #                 # "### Instruction:\n{instruction}\n\n### Response: "
    #                 if idx < len(dialog) - 1:
    #                     if idx % 2 == 0:
    #                         question += f"### Instruction:\n{value}\n\n"
    #                     else:
    #                         question += f"### Response: {value}\n\n"
    #                 else:
    #                     question += f"### Response: "
    #                     answer = value
    #         else:
    #             assert len(dialog) == 2
    #             question = dialog[0]['value']
    #             answer = dialog[1]['value']

    #         inputs = tokenizer(question, answer, return_tensors='pt').to("cuda")
    #         quality_score = rank_model(**inputs).logits[0].detach()

    #         processed += 1
    #         if processed >= 100:
    #             clean_memory()

    #     except Exception as e:
    #         print(dialog)
    #         print(e)
    #     return {'quality_score': quality_score}
    # dataset = dataset.map(_format, batched=False)

    # # def _format_batch(examples):
    # #     dialogs = examples['conversations']
    # #     questions = []
    # #     answers = []
    # #     for dialog in dialogs:
    # #         question = ""
    # #         answer = ""
    # #         if len(dialog) > 2:
    # #             for idx, round in enumerate(dialog):
    # #                 who = round['from']
    # #                 value = round['value']
    # #                 # "### Instruction:\n{instruction}\n\n### Response: "
    # #                 if idx < len(dialog) - 1:
    # #                     if idx % 2 == 0:
    # #                         question += f"### Instruction:\n{value}\n\n"
    # #                     else:
    # #                         question += f"### Response: {value}\n\n"
    # #                 else:
    # #                     question += f"### Response: "
    # #                     answer = value
    # #         else:
    # #             assert len(dialog) == 2
    # #             question = dialog[0]['value']
    # #             answer = dialog[1]['value']
    # #         questions.append(question)
    # #         answers.append(answer)
        
    # #     inputs = tokenizer(questions, answers, return_tensors='pt').to("cuda")
    # #     logits = rank_model(**inputs).logits
    # #     quality_scores = [l.detach() for l in logits]

    # #     clean_memory()

    # #     return {'quality_score': quality_scores}
    # # dataset = dataset.map(_format_batch, batched=True, batch_size=10)

    # dataset = dataset.sort('quality_score', reverse=True)

    # dataset.to_json(args.output_file, orient="records", lines=True, index=False)
    # print(f"Saved {len(dataset)} samples to {args.output_file}.")


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--reward_model_path", type=str, default="/opt/local/llm_models/huggingface.co/OpenAssistant/reward-model-deberta-v3-large-v2")
    parser.add_argument("--input_file", type=str, default="/opt/local/datasets/speechless_data/speechless-coding-16k.jsonl")
    parser.add_argument("--output_file", type=str, default="/opt/local/datasets/speechless_data/speechless-coding-16k-hqd.jsonl")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    main(args)