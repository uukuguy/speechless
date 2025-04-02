#!/bin/bash
# cat ~/openai_qwen2_7b_instruct.sh                                                       ─╯
# export OPENAI_API_PORT=30000
# export OPENAI_BASE_URL=http://localhost:${OPENAI_API_PORT}/v1
# export OPENAI_MODEL_NAME=/opt/local/llm_models/huggingface.co/Qwen/Qwen2.5-7B-Instruct
# export OPENAI_API_KEY=sk-unknown


SCRIPT_ROOT=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
PARENT_PATH=$(cd "${SCRIPT_PATH}/.." ; pwd)

if [ -f "${SCRIPT_ROOT}/task.env" ]; then
    echo "Loading task.env from ${SCRIPT_ROOT}"
    source ${SCRIPT_ROOT}/task.env
else
    echo "task.env not found in ${SCRIPT_ROOT}"
fi

MODEL_PATH=${TEST_MODEL_PATH:-${OPENAI_MODEL_NAME}}
SGLANG_PORT=${SGLANG_PORT:-${OPENAI_API_PORT}}
echo "MODEL_PATH: ${MODEL_PATH}"
echo "SGLANG_PORT: ${SGLANG_PORT}"

echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
GPUS=(${CUDA_VISIBLE_DEVICES//,/ })
NUM_GPUS=${#GPUS[@]}
if [ $NUM_GPUS -eq 0 ]; then
    NUM_GPUS=1
fi
echo "NUM_GPUS: $NUM_GPUS"

SHM_SIZE=${SHM_SIZE:-32g}

docker run --gpus all --rm \
    --privileged \
    --shm-size ${SHM_SIZE} \
    -p ${SGLANG_PORT}:${SGLANG_PORT} \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v /opt/local/llm_models:/opt/local/llm_models \
    --env "HF_TOKEN=${HF_TOKEN}" \
    --ipc=host \
    lmsysorg/sglang:latest \
    python3 -m sglang.launch_server \
    --model-path ${MODEL_PATH} \
    --host 0.0.0.0 \
    --port ${SGLANG_PORT} \
    --tensor-parallel-size ${NUM_GPUS} \
    $@
