#!/bin/bash

# after activating env and setting up accelerate...
langs=(py js java cpp swift php d jl lua r rkt rs)

model_path=$1
model=$(basename ${model_path})
org=speechlessai

PRECISION=bf16

for lang in "${langs[@]}"; do
    # use humaneval for py and multipl-e for the rest
    if [ "$lang" == "py" ]; then
        task=humaneval
    else
        task=multiple-$lang
    fi

    echo "Running task $task"
    generations_path=eval_results/bigcode_eval/generations_$model/generations_$task\_$model.json
    mkdir -p ${generations_path}
    accelerate launch \
        --num_processes=2 \
        --num_machines=1 \
        --mixed_precision=${PRECISION} \
        --dynamo_backend=no \
        eval/bigcode-evaluation-harness/main.py \
            --model ${model_path} \
            --task $task \
            --n_samples 50 \
            --batch_size 50 \
            --max_length_generation 512 \
            --temperature 0.2 \
            --precision bf16 \
            --trust_remote_code \
            --use_auth_token \
            --generation_only \
            --save_generations_path $generations_path
    echo "Task $task done"
done