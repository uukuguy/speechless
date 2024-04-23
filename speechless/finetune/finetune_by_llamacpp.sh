#!/bin/bash


    # --use-flash \

${HOME}/llama.cpp/finetune \
    --model-base /opt/local/llm_models/GGUF/7B/Meta-Llama-3-8B.Q8_0.gguf \
    --train-data /opt/local/datasets/alpaca_gpt4/alpaca_gpt4_llamacpp_data.txt \
    --lora-out /opt/local/llm_models/GGUF/7B/lora.gguf \
    -ngl 64 \
    --save-every 0 \
    --threads 8 \
    --ctx 2048 \
    --rope-freq-base 1000000 \
    --rope-freq-scale 1.0 \
    --batch 32 \
    --grad-acc 1 \
    --warmup 20 \
    --epochs 1 \
    --adam-iter 256000 \
    --adam-alpha 0.0002 \
    --adam-alpha 0.001 \
    --lora-r 32 \
    --lora-alpha 64 \
    --use-checkpointing \
    --sample-start "<s>" \
    --escape \
    --include-sample-start \
    --seed 18341 \
    --use-flash \