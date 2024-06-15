#!/usr/bin/env python
import torch
import os, sys, json
from tqdm import tqdm
from copy import deepcopy

SPEECHLESS_ROOT = os.getenv("SPEECHLESS_ROOT")
if SPEECHLESS_ROOT:
    sys.path.append(SPEECHLESS_ROOT)
else:
    sys.path.append("../../speechless")
from speechless.generate.llm import VllmAIModel

model_path = "/opt/local/llm_models/huggingface.co/01-ai/Yi-1.5-9B"
output_file = "digital_finance_test_result_Yi-1.5-9B.jsonl"
# model_path = "/opt/local/llm_models/huggingface.co/Tongyi-Finance-14B"
# output_file = "digital_finance_test_result_Tongyi_Finance_14B.jsonl"
test_file = "/opt/local/datasets/competitions/CCKS2024/digital_finance_test_prompts.jsonl"

test_data = [json.loads(line.strip()) for line in open(test_file).readlines()]
raw_datas = [ json.loads(line.strip()) for line in open("/opt/local/datasets/competitions/CCKS2024/digital_finance_test_data.jsonl").readlines()]
model = VllmAIModel(model_path=model_path, max_tokens=512)

max_examples = torch.cuda.device_count() * 4 

prompts = [data['instruction'] for data in test_data]
sampling_params = {
    'temperature': 0.1,
    'max_tokens': 512,
    # 'use_beam_search': False,
    # 'repetition_penalty': 1.0,
    # 'length_penalty': 1.0,
    # 'top_p': 1.0,
    # 'top_k':  -1,
    # 'seed': None,
    # 'stop': = None,
    # 'skip_special_tokens': True,
}
with open(output_file, 'w') as fd:
    for s, e, batch_responses in model.generate_batch(prompts, batch_size=max_examples, **sampling_params):
        assert e - s == len(batch_responses), f"{s=}, {e=}, {len(batch_responses)=}"
        for response, instruction, raw_data in zip(batch_responses, prompts[s:e], raw_datas[s:e]):
            raw_data = deepcopy(raw_data)
            raw_data['answer'] = response[:1] if 'A' in raw_data else response
            fd.write(json.dumps(raw_data, ensure_ascii=False) + "\n")
            fd.flush()

print(f"Saved {len(raw_datas)} lines in {output_file}")

# with open(output_file, 'w') as fd:
#     cached_examples = []
#     for i, (data, raw_data) in enumerate(tqdm(zip(test_data, raw_datas), ncols=100)):
#         if len(cached_examples) < max_examples:
#             cached_examples.append((data, raw_data))
#             if i < len(test_data) - 1:
#                 continue 

#         prompts = [data['instruction'] for data, _ in cached_examples]
#         sampling_params = {
#             'temperature': 0.1,
#         }
#         responses = model.generate(prompts, **sampling_params).strip()
#         for response, (data, raw_data) in zip(responses, cached_examples):
#             # -------- Generate responses --------
#             instruction = data['instruction']
#             print(f"{instruction=}")
#             print(f"{response=}")

#             raw_data['answer'] = response[:1] if 'A' in raw_data else response
#             fd.write(json.dumps(raw_data, ensure_ascii=False) + "\n")
#             fd.flush()

#         cached_examples = []

#         # -------- Generate responses --------
#         # instruction = data['instruction']
#         # response = model.generate(instruction, temperature=0.1).strip()
#         # print(f"{instruction=}")
#         # print(f"{response=}")

#         # raw_data['answer'] = response[:1] if 'A' in raw_data else response
#         # fd.write(json.dumps(raw_data, ensure_ascii=False) + "\n")
#         # fd.flush()

# print(f"Saved {len(raw_datas)} lines in {output_file}")
