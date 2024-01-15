#!/bin/bash

# MODEL_BASE_NAME=speechless-mistral-moloras-7b
# ORG_NAME=uukuguy
MODEL_BASE_NAME=speechless-nl2sql-ds-6.7b
ORG_NAME=uukuguy

LLAMA_CPP_ROOT=${HOME}/llama.cpp
MODEL_PATH=/opt/local/llm_models/huggingface.co/${ORG_NAME}/${MODEL_BASE_NAME}
OUTPUT_TYPE=f16
GGML_FILE=${MODEL_PATH}/ggml-model-${OUTPUT_TYPE}.gguf
GGUF_DIR=${MODEL_PATH}/GGUF

Q4_K_M_FILE=${GGUF_DIR}/${MODEL_BASE_NAME}.Q4_K_M.gguf
Q5_K_M_FILE=${GGUF_DIR}/${MODEL_BASE_NAME}.Q5_K_M.gguf
Q8_0_FILE=${GGUF_DIR}/${MODEL_BASE_NAME}.Q8_0.gguf

cd ${LLAMA_CPP_ROOT} && \
python convert.py \
    ${MODEL_PATH} \
    --padvocab \
    --outtype ${OUTPUT_TYPE} \
    --outfile ${GGML_FILE} && \
mkdir -p ${GGUF_DIR} && \
./quantize ${GGML_FILE} ${Q4_K_M_FILE} q4_k_m && \
./quantize ${GGML_FILE} ${Q5_K_M_FILE} q5_k_m && \
./quantize ${GGML_FILE} ${Q8_0_FILE} q8_0 && \
rm -f ${GGML_FILE} && \
ls -lh ${GGUF_DIR}
    
