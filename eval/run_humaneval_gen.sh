#!/bin/bash
# """
# git clone https://github.com/openai/human-eval.git && \
#    cd human-eval && pip install -v -e . && \
#    which evaluate_functional_correctness
#"""

model=$1
output_file=$2
output_path=$(dirname ${output_file})

# model="/opt/local/llm_models/huggingface.co/speechlessai/speechless-codellama-dolphin-orca-platypus-13b"
# model="/opt/local/llm_models/huggingface.co/speechlessai/speechless-codellama-airoboros-orca-platypus-13b.local"
# model="/opt/local/llm_models/huggingface.co/speechlessai/speechless-baichuan2-dolphin-orca-platypus-13b"
# model="/opt/local/llm_models/huggingface.co/speechlessai/speechless-codellama-34b-v1.0"
# model="/opt/local/llm_models/huggingface.co/TheBloke/Phind-CodeLlama-34B-v2-GPTQ"
# model="/opt/local/llm_models/huggingface.co/codellama/CodeLlama-13b-hf"
# model="/opt/local/llm_models/huggingface.co/microsoft/phi-1_5"

temp=0.2
max_len=2048
pred_num=1
num_seqs_per_iter=1

# temp=0.2
# max_len=512
# pred_num=50
# num_seqs_per_iter=50

# output_path=eval_results/human_eval/$(basename ${model})
# mkdir -p ${output_path}
# echo 'Output path: '$output_path
# echo 'Model to eval: '$model

# python eval/humaneval_gen.py \
#     --model ${model} \
#     --start_index 0 \
#     --end_index 164 \
#     --greedy_decode \
#     --temperature ${temp} \
#     --num_seqs_per_iter ${num_seqs_per_iter} \
#     --N ${pred_num} \
#     --max_len ${max_len} \
#     --output_path ${output_path} 


mkdir -p ${output_path}
echo 'Output file: '$output_file
echo 'Model to eval: '$model
PYTHONPATH=${SPEECHLESS_ROOT} \
python eval/humaneval_gen_vllm.py \
    --model ${model} \
    --start_index 0 \
    --end_index 164 \
    --temperature ${temp} \
    --num_seqs_per_iter ${num_seqs_per_iter} \
    --N ${pred_num} \
    --max_len ${max_len} \
    --sliding_window 0 \
    --output_file ${output_file} 