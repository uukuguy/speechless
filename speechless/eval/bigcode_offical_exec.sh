#!/bin/bash
# https://github.com/bigcode-project/bigcode-evaluation-harness/tree/main/leaderboard

langs=(py js java cpp swift php d jl lua r rkt rs)

model=$1
org=speechlessai

# if you provide absolute paths remove the $(pwd) from the command below
generations_path=generations_$model
metrics_path=metrics_$model

for lang in "${langs[@]}"; do
    if [ "$lang" == "py" ]; then
        task=humaneval
    else
        task=multiple-$lang
    fi

    gen_suffix=generations_$task\_$model.json
    metric_suffix=metrics_$task\_$model.json
    echo "Evaluation of $model on $task benchmark, data in $generations_path/$gen_suffix"

    sudo docker run -v $(pwd)/$generations_path/$gen_suffix:/app/$gen_suffix:ro  -v $(pwd)/$metrics_path:/app/$metrics_path -it evaluation-harness-multiple python3 main.py \
        --model $org/$model \
        --tasks $task \
        --load_generations_path /app/$gen_suffix \
        --metric_output_path /app/$metrics_path/$metric_suffix \
        --allow_code_execution  \
        --use_auth_token \
        --temperature 0.2 \
        --n_samples 50 | tee -a logs_$model.txt
    echo "Task $task done, metric saved at $metrics_path/$metric_suffix"
done