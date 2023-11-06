#!/bin/bash

cd tooleval 
export TOOLBENCH_DATA_DIR=/opt/local/datasets/toolbench_data
export CONVERTED_ANSWER_PATH=${TOOLBENCH_DATA_DIR}/reproduction_data/model_predictions_converted/ 
export SAVE_PATH=pass_rate_results 
export CANDIDATE_MODEL=toolllama_dfs
python eval_pass_rate.py \
    --converted_answer_path ${CONVERTED_ANSWER_PATH} \
    --save_path ${SAVE_PATH} \
    --reference_model ${CANDIDATE_MODEL} \
    --test_ids ${TOOLBENCH_DATA_DIR}/test_query_ids/ \
    --max_eval_threads 10 \
    --evaluate_times 4