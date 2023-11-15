#!/bin/bash

cd tooleval
export TOOLBENCH_DATA_DIR=/opt/local/datasets/toolbench_data
export RAW_ANSWER_PATH=${TOOLBENCH_DATA_DIR}/reproduction_data/model_predictions/
export CONVERTED_ANSWER_PATH=${TOOLBENCH_DATA_DIR}/reproduction_data/model_predictions_converted/
# export MODEL_NAME=speechless-tools-7b-32k-v0.15_dfs
# export MODEL_NAME=speechless-tools-7b-32k-v0.5-2871-steps
# export MODEL_NAME=speechless-tools-7b-32k-v0.5-5742-steps
# export MODEL_NAME=speechless-tools-7b-32k-mistral-2871-steps
#export MODEL_NAME=speechless-tools-7b-32k-mistral-5742-steps
export MODEL_NAME=speechless-tools-7b-32k-tora-3e-1435-steps
export METHOD=DFS_woFilter_w2
mkdir ${CONVERTED_ANSWER_PATH}/${MODEL_NAME}
for test_set in G1_instruction G1_category G1_tool G2_category G2_instruction G3_instruction
# for test_set in G1_instruction G1_category G1_tool G3_instruction
do
    answer_dir=${RAW_ANSWER_PATH}/${MODEL_NAME}/${test_set}
    output_file=${CONVERTED_ANSWER_PATH}/${MODEL_NAME}/${test_set}.json
    python convert_to_answer_format.py\
        --answer_dir ${answer_dir} \
        --method ${METHOD} \
        --output ${output_file}
done
