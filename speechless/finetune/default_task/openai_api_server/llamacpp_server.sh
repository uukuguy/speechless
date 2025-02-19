#!/bin/bash

# Assign the first argument to a variable text, if it is not empty, otherwise assign a default value
text=${1:-Which is bigger, 9.9 or 9}

# MODEL=/data01/DeepSeek-R1-GGUF/DeepSeek-R1-UD-Q2_K_XL.gguf
DEFAULT_MODEL=/data01/DeepSeek-R1-GGUF/DeepSeek-R1-UD-Q2_K_XL-00001-of-00005.gguf
MODEL=${OPENAI_DEFAULT_MODEL:-$DEFAULT_MODEL}

        # --no-kv-offload \
        # --tensor-split 0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125 \

llama-server --model ${MODEL} \
        --port 17080 \
        --ctx-size 8192 \
        --n-gpu-layers 62 \
        --threads 32 \
        --flash-attn 