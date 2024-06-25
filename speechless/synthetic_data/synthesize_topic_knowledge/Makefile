LLM_GENERATED_SUB_TOPICS_FILE=gdc-sub-topics-llm-generated.jsonl
SUB_TOPICS_FILE=gdc-sub-topics.jsonl
LLM_GENERATED_TOPIC_QUESTIONS_FILE=gdc-topic-questions-llm-generated.jsonl
TOPIC_QUESTIONS_FILE=gdc-topic-questions.jsonl
TOPIC_ANSWERS_FILE=gdc-topic-answers.jsonl

help:
	@echo "generate_sub_topics - Generate sub-topics for the GDC task"
	@echo "clean_sub_topics - Clean the generated sub-topics for the GDC task"

do_sub_topics:
	python -m speechless.synthetic_data.synthesize_topic_knowledge \
		--do_sub_topics \
		--output_file ${LLM_GENERATED_SUB_TOPICS_FILE} 

do_clean_sub_topics:
	python -m speechless.synthetic_data.synthesize_topic_knowledge \
		--do_clean_sub_topics \
		--input_file ${LLM_GENERATED_SUB_TOPICS_FILE} \
		--output_file ${SUB_TOPICS_FILE}

do_topic_questions:
	python -m speechless.synthetic_data.synthesize_topic_knowledge \
		--do_topic_questions \
		--input_file ${SUB_TOPICS_FILE} \
		--output_file ${LLM_GENERATED_TOPIC_QUESTIONS_FILE}

do_clean_topic_questions:
	python -m speechless.synthetic_data.synthesize_topic_knowledge \
		--do_clean_topic_questions \
		--input_file ${LLM_GENERATED_TOPIC_QUESTIONS_FILE} \
		--output_file ${TOPIC_QUESTIONS_FILE}

do_answer_questions:
	python -m speechless.synthetic_data.synthesize_topic_knowledge \
		--do_answer_questions \
		--key_id ${KEY_ID} \
		--start_idx ${START_IDX} \
		--end_idx ${END_IDX}
		--max_tokens 2048 \
		--input_file ${TOPIC_QUESTIONS_FILE} \
		--output_file gdc-topic-answers_${KEY_ID}_${START_IDX}_${END_IDX}.jsonl

answer_questions_w1:
	make do_answer_questions KEY_ID=1 START_IDX=0 END_IDX=500

answer_questions_w2:
	make do_answer_questions KEY_ID=1 START_IDX=500 END_IDX=1000

answer_questions_w3:
	make do_answer_questions KEY_ID=1 START_IDX=1000 END_IDX=1500

answer_questions_w4:
	make do_answer_questions KEY_ID=1 START_IDX=1000 END_IDX=2000

answer_questions_w5:
	make do_answer_questions KEY_ID=1 START_IDX=2000 END_IDX=2500