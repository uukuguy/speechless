#!/usr/bin/env bash
export PYTHONPATH=${SPEECHLESS_ROOT} && \

export GGUF_FILE=/opt/local/llm_models/embeddings/BAAI/bge-m3-fp16.gguf && \

# # the draft.gguf model should be a small variant of the target model.gguf
# llama-server -m model.gguf -md draft.gguf
# # use the /embedding endpoint
# llama-server -m model.gguf --embedding --pooling cls -ub 8192
# # use the /reranking endpoint
# llama-server -m model.gguf --reranking

llama-server --model ${GGUF_FILE} \
    --host 0.0.0.0 \
    --port 8030 \
    -np 8 \
    --n-gpu-layers 100 \
    --ctx-size 0 \
    --temp 0.6 \
    --min-p 0.2 \
