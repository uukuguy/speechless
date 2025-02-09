help:
	@echo "Usage: make [run]"

QUERY="损失函数"
REVIEW_TYPE="concept"

# QUERY="Text2SQL研究现状如何，面临哪些挑战？"
# REVIEW_TYPE="status"

# QUERY="有哪些方法可以提升大模型的规划能力，各自优劣是什么？"
# REVIEW_TYPE="comparison"

# QUERY="多模态大模型的技术发展路线是什么样的？"
# REVIEW_TYPE="timeline"

retrieve_papers:
	python retrieve_papers.py --query ${QUERY}

review_outlines:
	python review_outlines.py --query ${QUERY} --review_type ${REVIEW_TYPE}

knowledge_extraction_generate_prompts:
	python knowledge_extraction.py --generate_prompts --query ${QUERY} --max_tokens 8192

knowledge_extraction_answer_questions:
	python knowledge_extraction.py --answer_questions --query ${QUERY} \
		--num_processes 8 --pool_map_chunk_size 1

relevance_check_prompts:
	python answer_with_relevance_check.py --generate_prompts --query ${QUERY} 

relevance_check_answer_questions:
	python answer_with_relevance_check.py --answer_questions --query ${QUERY} \
		--num_processes 8 --pool_map_chunk_size 1

generate_paragraph_of_summary:
	python generate_paragraph_of_summary.py --query ${QUERY} --review_type ${REVIEW_TYPE}

demo1:
	python literature_review.py \
		--do_summary \
		--query "损失函数" \
		--review_type "concept" \

demo2:
	python literature_review.py \
		--do_summary \
		--query "Text2SQL研究现状如何，面临哪些挑战？" \
		--review_type "status" \

demo3:
	python literature_review.py \
		--do_summary \
		--query "有哪些方法可以提升大模型的规划能力，各自优劣是什么？" \
		--review_type "comparison" \

demo4:
	python literature_review.py \
		--do_summary \
		--query "多模态大模型的技术发展路线是什么样的？" \
		--review_type "timeline" \

run:
	python literature_review.py
