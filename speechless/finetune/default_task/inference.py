#!/usr/bin/env python
import os, sys, json
from tqdm import tqdm

SPEECHLESS_ROOT = os.getenv("SPEECHLESS_ROOT")
if SPEECHLESS_ROOT:
    sys.path.append(SPEECHLESS_ROOT)
else:
    sys.path.append("../../speechless")
from speechless.generate.llm import VllmAIModel

model_path = "/opt/local/llm_models/huggingface.co/speechlessai/AFAC2024-Track2-Yi-1.5-9B-2e"
test_file = "/opt/local/datasets/competitions/AFAC2024/AFAC2024-testing-data.jsonl"
output_file = "test_result_Yi-1.5-9B_2e.json"

test_data = [json.loads(line.strip()) for line in open(test_file).readlines()]
raw_data = json.load(open("/opt/local/datasets/competitions/AFAC2024/round1_training_data/test.json"))
model = VllmAIModel(model_path=model_path, max_tokens=4096)

with open(output_file, 'w') as fd:
    for data, final_data in tqdm(zip(test_data, raw_data), ncols=100):
        instruction = data['instruction']
        response = model.generate(instruction).strip()
        print(f"{instruction=}")
        print(f"{response=}")
        item = {
            'ID': final_data['ID'],
            'question': final_data['问题'],
            'answer': response
        }
        line = json.dumps(item, ensure_ascii=False)
        fd.write(line + "\n")
        fd.flush()

print(f"Saved {len(raw_data)} lines in {output_file}")
