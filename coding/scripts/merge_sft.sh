# From speechless-code-mistral-7b-v1.0
SCRIPT_ROOT=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
PARENT_PATH=$(cd "${SCRIPT_PATH}/.." ; pwd)

BASE_MODEL_PATH=Qwen3-4B
CHECKPOINT_DIR=adapter_model
TEST_MODEL_PATH=model_weights


python -m speechless.finetune.default_task.merge_peft_adapters \
    --base_model_name_or_path ${BASE_MODEL_PATH} \
    --lora_model_path ${CHECKPOINT_DIR} \
    --merged_model_name_or_path ${TEST_MODEL_PATH} \
    ${ADD_REASONING_TOKENS}