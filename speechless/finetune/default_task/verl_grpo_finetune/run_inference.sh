#!/usr/bin/env bash
SCRIPT_ROOT=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
PARENT_PATH=$(cd "${SCRIPT_PATH}/.." ; pwd)

source ${SCRIPT_ROOT}/task.env
unset http_proxy

TEST_FILE="test_data.jsonl"
# replace ".jsonl" with "-inferred.jsonl"
OUTPUT_FILE="${TEST_FILE%.jsonl}-inferred.jsonl"
LOGITS_PROCESSOR_MODULE_FILE="${PWD}/logits_processor.py"
LOGITS_PROCESSOR_CLASS_NAME=BacktickConstraint

PYTHONPATH=${SPEECHLESS_ROOT:-${HOME}/sandbox/LLM/speechless.ai/speechless} \
python -m speechless.finetune.default_task.inference_openai \
        --model_name ${TEST_MODEL_PATH} \
        --test_file ${TEST_FILE} \
        --output_file ${OUTPUT_FILE} \
        --base_url http://localhost:${VLLM_PORT}/v1 \
        --parallel_processes 4 \
        --parallel_chunk_size 16 \
        --request_batch_size 64 \
        --temperature 1.0 \
        --min_p 0.2 \
        --max_tokens 512 \
        --logits_processor_module_file ${LOGITS_PROCESSOR_MODULE_FILE} \
        --logits_processor_class_name ${LOGITS_PROCESSOR_CLASS_NAME} \
        --use_chat_template auto \
        --verbose

# python ${SCRIPT_ROOT}/inference_vllm.py \
#     --model_path ${TEST_MODEL_PATH} \
#     --test_file ${TEST_FILE} \
#     --output_file ${OUTPUT_FILE} \
#     --tensor_parallel_size ${TENSOR_PARALLEL_SIZE} \
#     --batch_size $((${TENSOR_PARALLEL_SIZE} * 4)) \
#     --temperature 0.1 \
#     --max_tokens 2048 

