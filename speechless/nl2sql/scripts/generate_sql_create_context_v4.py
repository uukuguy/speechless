#!/usr/bin/env python

import json
from tqdm import tqdm

raw_file = "../data/sql_create_context_v4.json"
json_data = json.load(open(raw_file))

instruction_template = open("../sql-eval/prompts/nl2sql_instruction.md").read()
input_template = open("../sql-eval/prompts/nl2sql_input.md").read()
response_template = open("../sql-eval/prompts/nl2sql_response.md").read()

final_data = []
for d in tqdm(json_data):
    question = d['question']
    context = d['context']
    answer = d['answer']

    instruction = instruction_template
    input = input_template.format(user_question=question, table_metadata_string=context)
    response = response_template.format(response=answer)
    question_data = {
        'instruction': instruction,
        'input': input,
        'response': response,
        "category": "sql_create_context_v4",
        "skip_prompt_formatting": False,
        "system": ""
    }

    final_data.append(question_data)

final_file = "../data/speechless-sql_create_context_v4.jsonl"
with open (final_file, 'w') as fd:
    for d in final_data:
        fd.write(f"{json.dumps(d, ensure_ascii=False)}\n")
print(f"Saved {len(final_data)} samples in {final_file}")