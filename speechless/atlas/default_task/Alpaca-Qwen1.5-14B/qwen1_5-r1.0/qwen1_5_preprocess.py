# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""
transform dataset to mindrecord.
"""
import argparse
import json
import os

import numpy as np

from mindspore.mindrecord import FileWriter

from qwen1_5.qwen1_5_tokenizer import Qwen2Tokenizer

IGNORE_TOKEN_ID = -100


def preprocess(messages, tokenizer, seq_length):
    """Preprocesses the data for supervised fine-tuning."""

    texts = []
    for _, msg in enumerate(messages):
        texts.append(
            tokenizer.apply_chat_template(
                msg,
                tokenize=True,
                add_generation_prompt=False,
                padding=True,
                max_length=seq_length,
                truncation=True,
            )
        )
    input_ids = np.array(texts)
    target_ids = np.array(texts)
    target_ids[target_ids == tokenizer.pad_token_id] = IGNORE_TOKEN_ID
    attention_mask = np.where(input_ids == tokenizer.pad_token_id, 0, 1)

    return dict(
        input_ids=input_ids, target_ids=target_ids, attention_mask=attention_mask
    )


def tokenize_qa(tokenizer, file_path, seq_length):
    raw_data = []
    with open(file_path, "r") as f:
        for line in f:
            raw_data.append(json.loads(line))
    dataset_cls = SupervisedDataset(raw_data, tokenizer, seq_length)
    for i in range(len(dataset_cls)):
        yield dataset_cls[i]


class SupervisedDataset:
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer, seq_length):
        super(SupervisedDataset, self).__init__()

        messages = [example["messages"] for example in raw_data]
        data_dict = preprocess(messages, tokenizer, seq_length)

        self.input_ids = data_dict["input_ids"]
        self.target_ids = data_dict["target_ids"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(
            input_ids=self.input_ids[i],
            target_ids=self.target_ids[i],
            attention_mask=self.attention_mask[i]
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_glob', type=str, required=True)
    parser.add_argument('--output_file', type=str,
                        default='./alpaca-fastchat-qwen.mindrecord')
    parser.add_argument('--vocab_file', default='./vocab.json', type=str,
                        help='vocab_file path')
    parser.add_argument('--merges_file', default='./merges.txt', type=str,
                        help='merge_file path')
    parser.add_argument('--file_partition', type=int, default=1)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--seq_length', type=int, default=2048)
    args = parser.parse_args()

    out_dir, out_file = os.path.split(os.path.abspath(args.output_file))
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    schema = {'input_ids': {"type": "int32", "shape": [-1]},
              'target_ids': {"type": "int32", "shape": [-1]},
              "attention_mask": {"type": "int32", "shape": [-1]}
              }
    writer = FileWriter(file_name=args.output_file,
                        shard_num=args.file_partition)
    writer.add_schema(schema)
    writer.open_and_set_header()

    transforms_count = 0
    word_tokenizer = Qwen2Tokenizer(
        args.vocab_file, args.merges_file, add_bos_token=False, add_eos_token=False)

    for x in tokenize_qa(word_tokenizer, args.input_glob, args.seq_length + 1):
        transforms_count += 1
        writer.write_raw_data([x])
    print("Transformed {} records.".format(transforms_count))

    writer.commit()
    out_file = args.output_file
    if args.file_partition > 1:
        out_file += '0'
    print("Transform finished, output files refer: {}".format(out_file))
