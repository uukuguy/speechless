#!/bin/bash

WANDB_PROJECT=$(basename $PWD) PYTHONPATH=${SPEECHLESS_ROOT} \
  python rft_grpo_unsloth.py 
