#!/usr/bin/env bash
# From speechless-code-mistral-7b-v1.0
SCRIPT_PATH=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
PARENT_PATH=$(cd "${SCRIPT_PATH}/.." ; pwd)

source ${SCRIPT_PATH}/task.env

CHECKPOINT_DIR=${TASK_CHECKPOINT_DIR}/checkpoint-2122/adapter_model

python ${SCRIPT_PATH}/../../scripts/merge_peft_adapters.py \
    --base_model_name_or_path ${BASE_MODEL_PATH} \
    --peft_model_path ${CHECKPOINT_DIR} \
    --merged_model_name_or_path ${TEST_MODEL_PATH} \
(env-llm) root@autodl-container-a8e51198ae-9af3cba7:~/sandbox/LLM/speechless.ai/speechless/tasks/speechless-code-mistral-7b-v1.0# ls
merge_peft_adapters.sh  outputs  run_api_server.sh  run_finetune.sh  task.env
(env-llm) root@autodl-container-a8e51198ae-9af3cba7:~/sandbox/LLM/speechless.ai/speechless/tasks/speechless-code-mistral-7b-v1.0# cat run_api_server.sh
#!/bin/bash
SCRIPT_PATH=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
PARENT_PATH=$(cd "${SCRIPT_PATH}/.." ; pwd)

source task.env

echo "SCRIPT_PATH: ${SCRIPT_PATH}"

PYTHONPATH=${SCRIPT_PATH}/../../.. \
python ../../api/server.py \
        --model_name_or_path=${TEST_MODEL_PATH} \
        --model_family vllm