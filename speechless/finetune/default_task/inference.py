#!/usr/bin/env python
import torch
import os, sys, json
from tqdm import tqdm
from copy import deepcopy
from loguru import logger

SPEECHLESS_ROOT = os.getenv("SPEECHLESS_ROOT")
if SPEECHLESS_ROOT:
    sys.path.append(SPEECHLESS_ROOT)
else:
    sys.path.append("../../speechless")
from speechless.generate.llm import VllmAIModel, HFAIModel

model_path = "/opt/local/llm_models/huggingface.co/speechlessai/"
output_file = "output.jsonl"
test_file = "/opt/local/datasets/competitions/IndustryCorpus2024/finance2024_test_data.jsonl"

test_data = [json.loads(line.strip()) for line in open(test_file).readlines()]
# raw_datas = [ json.loads(line.strip()) for line in open("/opt/local/datasets/competitions/CCKS2024/digital_finance_test_data.jsonl").readlines()]

max_tokens = 2048
# # vllm
# model = VllmAIModel(model_path=model_path, max_tokens=max_tokens)
# sampling_params = {
#     'temperature': 0.1,
#     'max_tokens': max_tokens,
#     # 'use_beam_search': False,
#     # 'repetition_penalty': 1.0,
#     # 'length_penalty': 1.0,
#     # 'top_p': 1.0,
#     # 'top_k':  -1,
#     # 'seed': None,
#     # 'stop': = None,
#     # 'skip_special_tokens': True,
# }

# hf
gen_kwargs = {
    "temperature": 0.75, 
    "max_new_tokens": 2048,
    "do_sample": True,
    # "top_p": 0.9,
    "min_p": 0.1,
}
model = HFAIModel(model_path=model_path, max_tokens=max_tokens, bits=8, gen_kwargs=gen_kwargs)

max_examples = torch.cuda.device_count() * 4 

prompts = [data['query'] for data in test_data]
with open(output_file, 'w', encoding='utf-8') as fd:
    # for s, e, batch_responses in model.generate_batch(prompts, batch_size=max_examples, **sampling_params):
    for s, e, batch_responses in model.generate_batch(prompts, batch_size=max_examples):
        assert e - s == len(batch_responses), f"{s=}, {e=}, {len(batch_responses)=}"
        # for response, instruction, raw_data in zip(batch_responses, prompts[s:e], raw_datas[s:e]):
        for response, instruction in zip(batch_responses, prompts[s:e]):
            logger.debug(f"{instruction=}")
            logger.info(f"{response=}")
            # raw_data = deepcopy(raw_data)
            # raw_data['answer'] = response[:1] if 'A' in raw_data else response
            # fd.write(json.dumps(raw_data, ensure_ascii=False) + "\n")
            # fd.flush()

print(f"Saved {len(test_data)} lines in {output_file}")
