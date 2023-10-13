LLM_MODELS=/opt/local/llm_models/huggingface.co

# BASE_MODEL=${LLM_MODELS}/meta-llama/Llama-2-13b-hf
# SUB_MODEL_1=${LLM_MODELS}/AIDC-ai-business/Luban-13B
# SUB_MODEL_2=${LLM_MODELS}/Open-Orca/OpenOrca-Platypus2-13B

# TARGET_MODEL=${LLM_MODELS}/uukuguy/speechless-llama2-luban-orca-platypus-13b

merge_models:
	PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
	python ties_merge.py \
		${BASE_MODEL} \
		${TARGET_MODEL} \
		--merge ${SUB_MODEL_1} \
		--merge ${SUB_MODEL_2} \
		--cuda

BASE_MODEL=${LLM_MODELS}/mistralai/Mistral-7B-v0.1
SUB_MODEL_1=${LLM_MODELS}/prepare_merge_mistral/dolphin-2.1-mistral-7b
SUB_MODEL_2=${LLM_MODELS}/prepare_merge_mistral/Mistral-7B-OpenOrca
SUB_MODEL_3=${LLM_MODELS}/prepare_merge_mistral/mistral-7b-platypus-fp16
SUB_MODEL_4=${LLM_MODELS}/prepare_merge_mistral/samantha-1.2-mistral-7b

TARGET_MODEL=${LLM_MODELS}/speechlessai/speechless-mistral-dolphin-orca-platypus-samantha-7b

merge_speechless_mistral:
	PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
	python ties_merge.py \
		${BASE_MODEL} \
		${TARGET_MODEL} \
		--merge ${SUB_MODEL_1} \
		--merge ${SUB_MODEL_2} \
		--merge ${SUB_MODEL_3} \
		--merge ${SUB_MODEL_4} \
		--cuda