#!/usr/bin/env bash
export PYTHONPATH=${SPEECHLESS_ROOT} && \

export GGUF_FILE=/opt/local/llm_models/huggingface.co/Qwen/Qwen3-4B-Q8_0.gguf && \

# # the draft.gguf model should be a small variant of the target model.gguf
# llama-server -m model.gguf -md draft.gguf
# # use the /embedding endpoint
# llama-server -m model.gguf --embedding --pooling cls -ub 8192
# # use the /reranking endpoint
# llama-server -m model.gguf --reranking

llama-server --model ${GGUF_FILE} \
    --host 0.0.0.0 \
    --port 8089 \
    -np 8 \
    --n-gpu-layers 100 \
    --ctx-size 0 \
    --temp 0.7 \
    --top_p 0.8 \
    --top_k 20 \
    --min-p 0.0 \

echo "Basic web UI can be accessed via browser: http://localhost:8089"
echo "Chat completion endpoint: http://localhost:8089/v1/chat/completions"
