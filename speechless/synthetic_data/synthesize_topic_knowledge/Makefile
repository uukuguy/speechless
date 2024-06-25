LLM_GENERATED_SUB_TOPICS_FILE=gdc-sub-topics-llm-generated.jsonl
SUB_TOPICS_FILE=gdc-sub-topics.jsonl
LLM_GENERATED_TOPIC_QUESTIONS_FILE=gdc-topic-questions-llm-generated.jsonl

help:
	@echo "generate_sub_topics - Generate sub-topics for the GDC task"
	@echo "clean_sub_topics - Clean the generated sub-topics for the GDC task"

do_sub_topics:
	python speechless.synthetic_data.synthesize_topic_knowledge \
		--do_sub_topics \
		--output_file ${LLM_GENERATED_SUB_TOPICS_FILE} 

do_clean_sub_topics:
	python speechless.synthetic_data.synthesize_topic_knowledge \
		--do_clean_sub_topics \
		--input_file ${LLM_GENERATED_SUB_TOPICS_FILE} \
		--output_file ${SUB_TOPICS_FILE}

do_topic_questions:
	python synthesize_topic_knowledge.py \
		--do_topic_questions \
		--input_file ${SUB_TOPICS_FILE} \
		--output_file ${LLM_GENERATED_TOPIC_QUESTIONS_FILE}
