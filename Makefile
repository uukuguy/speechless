# include Makefile.env

MODELS_ROOT_DIR=/opt/local/llm_models/huggingface.co
# BASE_MODEL_PATH=${MODELS_ROOT_DIR}/Phind/Phind-CodeLlama-34B-v2
#BASE_MODEL_PATH=${MODELS_ROOT_DIR}/mistralai/Mistral-7B-v0.1
BASE_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-mistral-six-in-one-7b

# pass@1: 75.61
# TEST_MODEL_PATH=/opt/local/llm_models/huggingface.co/speechlessai/speechless-codellama-34b-v2.0
# pass@1: 70.73
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-codellama-34b-v1.9

# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-mistral-7b-v0.1
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-nl2sql-mistral-7b-v0.1
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/mistralai/Mistral-7B-v0.1
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/Phind/Phind-CodeLlama-34B-v2

# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-code-mistral-orca-7b-v1.0
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-code-mistral-7b-v1.0
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-tora-code-7b-v1.0
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-mistral-dolphin-orca-platypus-samantha-7b
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/uukuguy/speechless-mistral-six-in-one-7b
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-tora-code-7b-v1.0

# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/Open-Orca/Mistral-7B-OpenOrca
#TEST_MODEL_PATH=${MODELS_ROOT_DIR}/Mistral-7B-OpenOrca-lora-merged
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/Mistral-7B-OpenOrca-r256-lora-merged
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/Mistral-7B-OpenOrca-r128-lora-merged

# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-code-mistral-7b-v1.1
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-code-mistral-7b-v1.0
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-code-mistral-orca-7b-v1.0

# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-tora-code-7b-v1.0
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/mistralai/Mistral-7B-v0.1

# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-codellama-34b-v2.1
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-codellama-34b-v2.0
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-tora-code-7b-v1.0

# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-reasoning-7b-v0
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-reasoning-7b-v0.1-tora
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-reasoning-7b-v0.2-mistral

#TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-reasoning-7b-v0.2-tora
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-reasoning-7b-v0.2-tora-cosine
#TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-reasoning-7b-v0.3-tora
#TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-reasoning-7b-v0.4-tora
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-agents-7b-v0.1-tora
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-agents-7b-v0.1-32k-mistral
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-reasoning-7b-v0.3-mistral
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/EleutherAI/llemma_7b
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-agents-7b-v0.1-tora
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-agents-7b-v0.1-32k-mistral
#TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-agents-7b-v0.1-32k-tora
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-agents-7b-v0.2-32k-tora
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-agents-7b-v0.2-32k-tora-794-steps
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-agents-7b-v0.2-32k-mistral
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-agents-7b-v0.2.2-32k-tora

#TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-tora-code-7b-v1.0

# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/mistralai/Mistral-7B-v0.1
#TEST_MODEL_PATH=${MODELS_ROOT_DIR}/llm_agents/tora-code-7b-v1.0
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-mistral-orca-7b-16k
#TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechless-agents-7b-v0.3-32k-mistral
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-mistral-orca-7b-32k-644-steps
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/Phind/Phind-CodeLlama-34B-v2
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/stabilityai/stablelm-3b-4e1t
#TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-coding-7b-16k-tora-1357-steps
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-coding-7b-16k-tora-2714-steps
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-coding-7b-16k-mistral-2714-steps

# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-coding-7b-16k-v1.3-mistral-1357-steps
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-coding-7b-16k-v1.3-mistral-2715-steps
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-coding-7b-16k-v1.3-mistral-4071-steps

# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-agents-7b-v0.2-32k-mistral

# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-coding-7b-orca2-1357-steps
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-coding-7b-orca2-1e
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-mistral-7b-dare-0.85
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/mixture-of-multi-loras/Intel/neural-chat-7b-v3-1-dare-0.85
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/uukuguy/speechless-mistral-six-in-one-7b
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/uukuguy/Orca-2-13b-f16
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/uukuguy/Orca-2-7b-f16
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-coding-7b-orca2-1e
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-coding-7b-orca2-130-steps
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-coding-7b-16k-tora

# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/uukuguy/speechless-code-mistral-7b-v1.0
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-coding-7b-16k-tora

# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/llm-agents/tora-code-34b-v1.0
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-coding-34b-8k-tora-1500-steps
# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-coding-34b-16k-tora-640-steps

# TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-coding-34b-8k-tora-1500-steps

TEST_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-code-mistral-7b-v2.0-2628-steps

TASK_NAME=$(shell basename ${TEST_MODEL_PATH})

OUTPUT_DIR=./outputs
CHECKPOINT_DIR=${OUTPUT_DIR}/${TASK_NAME}/checkpoint-3500/adapter_model


help:
	@echo "Usage: make [prepare_data | finetune | inference | eval]" 

# -------------------- Train --------------------

prepare_data:
	python ./scripts/prepare_data.py

finetune_13b:
	bash ./scripts/finetune_speechless_codellam_13b.sh

finetune_34b:
	bash ./scripts/finetune_speechless_codellam_34b.sh

full_finetune_34b:
	bash ./scripts/full_finetune_speechless_codellam_34b.sh

finetune_v2.1:
	# bash ./scripts/finetune_speechless_codellam_34b_v2.1.sh
	cd tasks/speechless_codellama_34b_v2.1 && \
		bash ./finetune_speechless_codellama_34b_v2.1.sh

finetune_mistral_7b:
	# bash ./scripts/finetune_speechless_mistral_7b_v0.1.sh
	cd tasks/speechless_mistral_7b_v0.1 && \
		bash ./finetune_speechless_mistral_7b_v0.1.sh

merge_peft_adapters:
	PYTHONPATH=. \
	python scripts/merge_peft_adapters.py \
		--base_model_name_or_path ${BASE_MODEL_PATH} \
		--peft_model_path ${CHECKPOINT_DIR} \
		--merged_model_name_or_path ${TEST_MODEL_PATH} \


# -------------------- Inference --------------------

inference:
	PYTHONPATH=${SPEECHLESS_ROOT} \
	python inference.py \
		--base_model ${TEST_MODEL_PATH} \
		--test_file_path ${TEST_FILE} \

inference_with_lora:
	PYTHONPATH=${SPEECHLESS_ROOT} \
	python inference.py \
		--base_model ${BASE_MODEL_PATH} \
		--lora_weights ${CHECKPOINT_DIR} \
		--test_file_path ${TEST_FILE} \


# -------------------- lm-evaluation-harness --------------------
#https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard

lmeval:
	PYTHONPATH=${SPEECHLES_ROOT} \
	python -m speechless.eval.lmeval \
		--do_gen \
		--model hf-causal-experimental \
		--model_args pretrained=${TEST_MODEL_PATH},use_accelerate=True \
		--batch_size 4 \
		--write_out \
		--output_path eval_results/lm_eval/${TASK_NAME} 

lmeval_show_results:
	python -m speechless.eval.lmeval \
		--show_result \
		--output_path eval_results/lm_eval/${TASK_NAME} 


# -------------------- HumanEval --------------------
HUMANEVAL_OUTPUT_DIR=eval_results/human_eval/${TASK_NAME}

humaneval:
	PYTHONLIB=${SPEECHLESS_ROOT} \
	python -m speechless.eval.humaneval \
		--do_gen \
		--show_results \
		--model ${TEST_MODEL_PATH} \
        --output_dir ${HUMANEVAL_OUTPUT_DIR}

humaneval_chatlm:
	PYTHONLIB=${SPEECHLESS_ROOT} \
	python -m speechless.eval.humaneval \
		--do_gen \
		--show_results \
		--model ${TEST_MODEL_PATH} \
        --output_dir ${HUMANEVAL_OUTPUT_DIR} \
		--prompt_type chatlm

humaneval_show_results:
	PYTHONLIB=${SPEECHLESS_ROOT} \
	python -m speechless.eval.humaneval \
		--show_results \
		--model ${TEST_MODEL_PATH} \
        --output_dir ${HUMANEVAL_OUTPUT_DIR}


# -------------------- Big Code Evaluation Harness --------------------
bigcode_eval_gen:
	./speechless/eval/run_bigcode_eval_gen.sh ${TEST_MODEL_PATH}

bigcode_eval:
	./speechless/eval/run_bigcode_eval.sh ${TEST_MODEL_PATH} 


# -------------------- MultiPL-E --------------------
# https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard

multiple_gen:
	bash ./speechless/eval/run_multiple_gen.sh ${TEST_MODEL_PATH}

multiple_gen_50:
	bash speechless/eval/run_multiple_gen.sh ${TEST_MODEL_PATH} 50 20

multiple_50:
	bash speechless/eval/run_multiple.sh ${TEST_MODEL_PATH} 50

multiple:
	bash ./speechless/eval/run_multiple.sh ${TEST_MODEL_PATH}


# -------------------- speechless.api.server --------------------
api_server:
	PYTHONPATH=${SPEECHLESS_ROOT} \
	python -m speechless.api.server \
		--model_name_or_path ${TEST_MODEL_PATH} \
		--model_family vllm \

include ../Makefile.remote
