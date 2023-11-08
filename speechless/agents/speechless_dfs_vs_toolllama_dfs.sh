#!/bin/bash

cd tooleval
export TOOLBENCH_DATA_DIR=/opt/local/datasets/toolbench_data
export CONVERTED_ANSWER_PATH=${TOOLBENCH_DATA_DIR}/reproduction_data/model_predictions_converted/
export SAVE_PATH=preference_results
export PASS_TARE_PATH=pass_rate_results
export REFERENCE_MODEL=toolllama_dfs
# export CANDIDATE_MODEL=speechless-tools-7b-32k-v0.15_dfs
# export CANDIDATE_MODEL=speechless-tools-7b-32k-v0.5-2871-steps
export CANDIDATE_MODEL=speechless-tools-7b-32k-v0.5-5742-steps
# export API_POOL_FILE=path/to/your/openai_key_json_file.json
python eval_preference.py \
    --converted_answer_path ${CONVERTED_ANSWER_PATH} \
    --reference_model ${REFERENCE_MODEL} \
    --output_model ${CANDIDATE_MODEL} \
    --test_ids ${TOOLBENCH_DATA_DIR}/test_query_ids/ \
    --save_path ${SAVE_PATH} \
    --pass_rate_result_path ${PASS_TARE_PATH} \
    --max_eval_threads 10 \
    --use_pass_rate true \
    --evaluate_times 1
