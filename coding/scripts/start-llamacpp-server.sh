#!/usr/bin/env bash
export PYTHONPATH=${SPEECHLESS_ROOT} && \
export MODEL_NAME=DIFC2025-round_a-SFT-Qwen3-4B && \

export OUTTYPE=$@ && \
export OUTTYPE=${OUTTYPE:-q8_0} && \
# convert OUTTYPE to upper case
export NAME_OUTTYPE=$(echo ${OUTTYPE} | tr '[:lower:]' '[:upper:]')
export GGUF_FILE=${MODEL_NAME}-${NAME_OUTTYPE}.gguf && \

export GGUF_FILE=DIFC2025-round_a-SFT-Qwen3-4B-F16-00001-of-00002.gguf

# # the draft.gguf model should be a small variant of the target model.gguf
# llama-server -m model.gguf -md draft.gguf
# # use the /embedding endpoint
# llama-server -m model.gguf --embedding --pooling cls -ub 8192
# # use the /reranking endpoint
# llama-server -m model.gguf --reranking

# export CHAT_TEMPLATE=$(cat chat_template_Qwen3.jinja)

    # --jinja \
    # --chat-template "${CHAT_TEMPLATE}" \
    # --chat-template-file chat_template_Qwen3.jinja

llama-server --model ${GGUF_FILE} \
    --host 0.0.0.0 \
    --port 8089 \
    -np 8 \
    --n-gpu-layers 100 \
    --ctx-size 0 \
    --temp 0.6 \
    --min-p 0.2 \

echo "Basic web UI can be accessed via browser: http://localhost:8089"
echo "Chat completion endpoint: http://localhost:8089/v1/chat/completions"
