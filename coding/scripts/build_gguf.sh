#!/usr/bin/env bash
export PYTHONPATH=${SPEECHLESS_ROOT} && \
export MODEL_NAME=DIFC2025-round_a-SFT-Qwen3-4B && \

export OUTTYPE=$@ && \
export OUTTYPE=${OUTTYPE:-q8_0} && \
# convert OUTTYPE to upper case
export NAME_OUTTYPE=$(echo ${OUTTYPE} | tr '[:lower:]' '[:upper:]')
export GGUF_FILE=${MODEL_NAME}-${NAME_OUTTYPE}.gguf && \

echo "Converting ${MODEL_NAME} to GGUF format with output type ${OUTTYPE}..."

python -m  speechless.llamacpp.convert_hf_to_gguf \
    --outfile ${GGUF_FILE} \
    --model-name ${MODEL_NAME}  \
    --outtype ${OUTTYPE} \
    --split-max-size 5G \
    model_weights

echo "Conversion complete: ${GGUF_FILE}"
echo "To run the model, use the following command:"
echo "llama-cli serve ${GGUF_FILE} --n-gpu-layers 100 --gpu-memory-fraction 0.5 --n-ctx 2048 --n-batch 512 --n-gpu 1"
