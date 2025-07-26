#!/bin/bash

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir checkpoints/e3_4b/global_step_110/actor \
    --target_dir /opt/local/llm_models/huggingface.co/speechlessai/AFAC2025-round_3-RFT-Qwen3-4B