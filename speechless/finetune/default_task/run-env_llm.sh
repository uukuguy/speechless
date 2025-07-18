#!/bin/bash
# run-env_llm.sh -p 12345:12345 --gpus="all" --env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# run-env_llm.sh -p 12345:12345 --gpus="device=1" --env CUDA_VISIBLE_DEVICES=1

export TASK_NAME=$(basename $(pwd))
# export SL_VERSION=cuda125-ubuntu24.04
export SL_VERSION=cuda123-ubuntu22.04

#    --network host \
#    --name env-llm \

#    --privileged \

docker run -it --rm \
    --privileged \
    --shm-size=64G \
    -v ${PWD}:/workspace/${TASK_NAME} \
    -v ${HOME}/.cache:/root/.cache \
    -v /opt/local:/opt/local \
    -v ${HOME}/sandbox/LLM/speechless.ai/speechless:/root/sandbox/LLM/speechless.ai/speechless \
    -v ${HOME}/sandbox/LLM/speechless.ai/tasks:/root/sandbox/LLM/speechless.ai/tasks \
    -v ${HOME}/.zsh_history:/root/.zsh_history \
    -v ${HOME}/.tmux_env_llm/resurrect:/root/.tmux/resurrect \
    -v /data01:/data01 \
    -v ${HOME}/.netrc:/root/.netrc \
    --env BNB_CUDA_VERSION=123 \
    --env SHELL=/usr/bin/zsh \
    --env https_proxy=http://28.160.2.68:808 \
    --env WORK_DIR=/root/sandbox/LLM/speechless.ai/tasks/${TASK_NAME} \
    $@ \
    speechlessai/env-llm:${SL_VERSION}
