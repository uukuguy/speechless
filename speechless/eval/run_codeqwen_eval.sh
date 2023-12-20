#!/bin/bash
SCRIPT_PATH=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
PARENT_PATH=$(cd "${SCRIPT_PATH}/.." ; pwd)

TEST_MODEL_PATH=$1
TASK_NAME=$(basename ${TEST_MODEL_PATH})

# TASK=$2
# BIGCODE_TASKS="humaneval,multiple-java,multiple-js,multiple-cpp,multiple-rs,multiple-go,multiple-sh,multiple-jl,multiple-swift,multiple-php,multiple-d,multiple-lua,multiple-r,multiple-rkt"

# BIGCODE_TASKS="humaneval multiple-java multiple-js multiple-cpp multiple-rs multiple-jl multiple-swift multiple-php multiple-d multiple-lua multiple-r multiple-rkt"

# CODEQWEN_TASKS="humanevalsynthesize-python humanevalsynthesize-java humanevalsynthesize-js humanevalsynthesize-cpp humanevalsynthesize-go humanevalsynthesize-rust humanevalfixtests-python humanevalfixtests-java humanevalfixtests-js humanevalfixtests-cpp humanevalfixtests-go humanevalfixtests-rust mbpp"


# TEMPERATURE=0.2
# N_SAMPLES=1
# # LIMIT=100
# # --limit ${LIMIT} \

# for TASK in ${CODEQWEN_TASKS}; do
#     echo "Running ${task}"
#     TASK_RESULTS_DIR=${PWD}/eval_results/bigcode_eval/${TASK_NAME}
#     TASK_GENERATIONS_FILE=generations_${TASK}_${TASK_NAME}.json
#     TASK_GENERATIONS_PATH=${TASK_RESULTS_DIR}/${TASK_GENERATIONS_FILE} 
#     TASK_METRIC_RESULTS_FILE=generations_${TASK}_${TASK_NAME}_results.json
#     docker run -it --rm \
#         -v ${TASK_RESULTS_DIR}:/eval_results:rw \
#         code-qwen-competition \
#         python3 main.py \
#             --model ${TEST_MODEL_PATH} \
#             --tasks ${TASK} \
#             --load_generations_path /eval_results/${TASK_GENERATIONS_FILE} \
#             --metric_output_path /eval_results/${TASK_METRIC_RESULTS_FILE} \
#             --allow_code_execution  \
#             --temperature ${TEMPERATURE} \
#             --n_samples ${N_SAMPLES}
# done;


# please replace this with your own model name which is taken during generation with launch_generate_codeqwen.sh
# model={your-model-name}
# org=test

model=${TASK_NAME}


TASK_RESULTS_DIR=${PWD}/eval_results/bigcode_eval/${TASK_NAME}

#tasks=(humanevalsynthesize-python humanevalsynthesize-java humanevalsynthesize-js humanevalsynthesize-cpp humanevalsynthesize-go humanevalsynthesize-rust humanevalfixtests-python humanevalfixtests-java humanevalfixtests-js humanevalfixtests-cpp humanevalfixtests-go humanevalfixtests-rust mbpp)
#tasks=(humanevalsynthesize-python humanevalsynthesize-java humanevalsynthesize-js humanevalsynthesize-cpp humanevalsynthesize-rust humanevalfixtests-python humanevalfixtests-java humanevalfixtests-js humanevalfixtests-cpp humanevalfixtests-go humanevalfixtests-rust mbpp)
tasks=(humanevalsynthesize-go)

# if you provide absolute paths remove the $(pwd) from the command below
generations_path=${TASK_RESULTS_DIR}
metrics_path=metrics_$model

# if [ -d $metrics_path ]; then
#     echo "Folder exists. Deleting folder: $metrics_path"
#     rm -rf $metrics_path
# fi
mkdir -p $metrics_path

batch_size=1
n_samples=1
eos_token="\"<|im_end|>\""


for task in "${tasks[@]}"; do
    echo "Task: $task"

    gen_suffix=generations_$task\_$model.json
    metric_suffix=metrics_$task\_$model.json
    echo "Evaluation of $model on $task benchmark, data in $generations_path/$gen_suffix"

    docker run \
    -v $generations_path/$gen_suffix:/app/$gen_suffix:ro  \
    -v $(pwd)/$metrics_path:/app/$metrics_path \
    -it --rm code-qwen-competition \
    bash -c "python3 main.py \
        --model ${TEST_MODEL_PATH} \
        --tasks ${task} \
        --prompt instruct \
        --load_generations_path /app/$gen_suffix \
        --metric_output_path /app/$metrics_path/$metric_suffix \
        --allow_code_execution  \
        --trust_remote_code \
        --use_auth_token \
        --temperature 0.2 \
        --max_length_generation 1024 \
        --do_sample False \
        --precision bf16 \
        --eos "$eos_token" \
        --seed 999999999 \
        --batch_size $batch_size \
        --n_samples $n_samples | tee -a logs_$model.txt"
    echo "Task $task done, metric saved at $metrics_path/$metric_suffix"
done
