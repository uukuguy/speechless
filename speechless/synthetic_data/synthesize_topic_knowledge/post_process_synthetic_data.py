#!/usr/bin/env python

import json
from collections import defaultdict
from tqdm import tqdm

synthetic_data_file = "./gdc-topic-answers-13997.jsonl"
lines = open(synthetic_data_file).readlines()
json_data = [json.loads(line.strip()) for line in lines]

topic_question_answers = defaultdict(defaultdict(list))
num_question_answers = 0

for data in tqdm(json_data, ncols=100):
    main_topic = data['main_topic']
    sub_topic = data['sub_topic']
    questions = data['questions']
    qa_list = []
    for q in questions:
        if 'question' in q:
            question = q['question']
        elif 'request' in q:
            question = q['request']
        else:
            raise ValueError("No question or request found in the question data.")
        answer = q['answer']
        qa_list.append({'question': question, 'answer': answer})
    num_question_answers += len(qa_list)
    topic_question_answers[main_topic][sub_topic].extend(qa_list)

output_file = "./gdc-topic-answers-13997-processed.json"
json.dump(topic_question_answers, open(output_file, "w"), ensure_ascii=False, indent=2)
print(f"Saved the {num_question_answers} processed topic question answers to {output_file}")
