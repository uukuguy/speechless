#!/bin/bash
# """
# git clone https://github.com/openai/human-eval.git && \
#    cd human-eval && pip install -v -e . && \
#    which evaluate_functional_correctness
#"""

SCRIPT_PATH=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
PARENT_PATH=$(cd "${SCRIPT_PATH}/.." ; pwd)

model=$1
output_file=$2
output_path=$(dirname ${output_file})

temp=0.2
max_len=1024
gen_batch_size=16

echo 'Output file: '$output_file
echo 'Model to eval: '$model

# PYTHONPATH=${SPEECHLESS_ROOT} \
# python ${SCRIPT_PATH}/humaneval_gen_vllm.py \

PYTHONPATH=${SPEECHLESS_ROOT} \
python -m speechless.eval.humaneval_gen_vllm \
    --model ${model} \
    --gen_batch_size ${gen_batch_size} \
    --start_index 0 \
    --end_index 164 \
    --temperature ${temp} \
    --max_len ${max_len} \
    --output_file ${output_file} 