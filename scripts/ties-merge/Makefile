LLM_MODELS=/opt/local/llm_models/huggingface.co

# BASE_MODEL=${LLM_MODELS}/meta-llama/Llama-2-13b-hf
# SUB_MODEL_1=${LLM_MODELS}/AIDC-ai-business/Luban-13B
# SUB_MODEL_2=${LLM_MODELS}/Open-Orca/OpenOrca-Platypus2-13B

# TARGET_MODEL=${LLM_MODELS}/uukuguy/speechless-llama2-luban-orca-platypus-13b

# merge_models:
# 	PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
# 	python ties_merge.py \
# 		${BASE_MODEL} \
# 		${TARGET_MODEL} \
# 		--merge ${SUB_MODEL_1} \
# 		--merge ${SUB_MODEL_2} \
# 		--cuda

# ---------- speechless-mistral-dolphin-orca-platypus-samantha-7b ----------
# BASE_MODEL=${LLM_MODELS}/mistralai/Mistral-7B-v0.1
# SUB_MODEL_1=${LLM_MODELS}/prepare_merge_mistral/dolphin-2.1-mistral-7b
# SUB_MODEL_2=${LLM_MODELS}/prepare_merge_mistral/Mistral-7B-OpenOrca
# SUB_MODEL_3=${LLM_MODELS}/prepare_merge_mistral/mistral-7b-platypus-fp16
# SUB_MODEL_4=${LLM_MODELS}/prepare_merge_mistral/samantha-1.2-mistral-7b

# TARGET_MODEL=${LLM_MODELS}/speechlessai/speechless-mistral-dolphin-orca-platypus-samantha-7b

# ---------- speechless-six-in-one-7b ----------
# BASE_MODEL=${LLM_MODELS}/mistralai/Mistral-7B-v0.1
# SUB_MODEL_1=${LLM_MODELS}/prepare_merge_mistral/dolphin-2.1-mistral-7b
# SUB_MODEL_2=${LLM_MODELS}/prepare_merge_mistral/Mistral-7B-OpenOrca
# SUB_MODEL_3=${LLM_MODELS}/prepare_merge_mistral/mistral-7b-platypus-fp16
# SUB_MODEL_4=${LLM_MODELS}/prepare_merge_mistral/samantha-1.2-mistral-7b
# SUB_MODEL_5=${LLM_MODELS}/prepare_merge_mistral/CollectiveCognition-v1.1-Mistral-7B
# SUB_MODEL_6=${LLM_MODELS}/prepare_merge_mistral/zephyr-7b-alpha

# TARGET_MODEL=${LLM_MODELS}/speechlessai/speechless-six-in-one-7b

# ---------- speechless-mistral-7b-dare-0.85 ----------
# MIXTURE_OF_MUTLI_LORAS_DIR=/opt/local/llm_models/huggingface.co/mixture-of-multi-loras
# BASE_MODEL=${LLM_MODELS}/mistralai/Mistral-7B-v0.1
# SUB_MODEL_1=${MIXTURE_OF_MUTLI_LORAS_DIR}/Intel/neural-chat-7b-v3-1-dare-0.85
# SUB_MODEL_2=${MIXTURE_OF_MUTLI_LORAS_DIR}/bhenrym14/mistral-7b-platypus-fp16-dare-0.85
# SUB_MODEL_3=${MIXTURE_OF_MUTLI_LORAS_DIR}/jondurbin/airoboros-m-7b-3.1.2-dare-0.85
# SUB_MODEL_4=${MIXTURE_OF_MUTLI_LORAS_DIR}/migtissera/SynthIA-7B-v1.3-dare-0.85
# SUB_MODEL_5=${MIXTURE_OF_MUTLI_LORAS_DIR}/teknium/CollectiveCognition-v1.1-Mistral-7B-dare-0.85
# SUB_MODEL_6=${MIXTURE_OF_MUTLI_LORAS_DIR}/uukuguy/speechless-mistral-dolphin-orca-platypus-samantha-7b-dare-0.85


# ---------- speechless-six-in-one-7b ----------
MIXTURE_OF_MUTLI_LORAS_DIR=${LLM_MODELS}/mixture-of-multi-loras
BASE_MODEL=${LLM_MODELS}/mistralai/Mistral-7B-v0.1
SUB_MODEL_1=${LLM_MODELS}/uukuguy/zephyr-7b-beta-4bit-r64-lora-merged
SUB_MODEL_2=${LLM_MODELS}/uukuguy/speechless-code-mistral-7b-v1.0-4bit-r64-lora-merged
SUB_MODEL_3=${LLM_MODELS}/uukuguy/functionary-small-v2.2-4bit-r64-lora-merged

TARGET_MODEL=${MIXTURE_OF_MUTLI_LORAS_DIR}/uukuguy/speechless-zephyr-code-functionary-7b

merge_models:
	PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
	python ties_merge.py \
		${BASE_MODEL} \
		${TARGET_MODEL} \
		--merge ${SUB_MODEL_1} \
		--merge ${SUB_MODEL_2} \
		--merge ${SUB_MODEL_3} \
