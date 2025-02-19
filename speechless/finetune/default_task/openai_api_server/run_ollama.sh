#/bin/bash

# https://martech.org/how-to-run-deepseek-locally-on-your-computer/

    # --num-threads 32 \
    # --ngl 999 \
    # --context-size 8192 \
    # --temp 0.6 \

# ollama run -i DeepSeek-R1-UD-IQ1_M --verbose \
    
# https://blog.csdn.net/arkohut/article/details/139426912
# OLLAMA_MODELS=/root/autodl-tmp/models
OLLAMA_HOST=0.0.0.0:6006 \
OLLAMA_NUM_PARALLEL=16 \
OLLAMA_FLASH_ATTENTION=1 \
OLLAMA_KEEP_ALIVE=-1 \
OLLAMA_MAX_QUEUE=1000 \
    ollama serve

# /etc/systemd/system/ollama.service  
# [Service]
# Environment="OLLAMA_HOST=0.0.0.0:11434"
# Environment="OLLAMA_MODELS=/data/ollama/models"
# Environment="OLLAMA_NUM_PARALLEL=16"
# Environment="OLLAMA_FLASH_ATTENTION=1"
# Environment="OLLAMA_KEEP_ALIVE=-1"
# Environment="OLLAMA_MAX_QUEUE=1000"
# ExecStart=/usr/bin/ollama serve
