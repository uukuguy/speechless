# Speechless

## speechless-codellama-34b-v2.0

### HumanEval

humaneval-python pass@1: 75.61

### MultiPL-E

| | |
| ------ | ------ |
| python | 67.55 |
| java | 51.93 |
| javascript | 64.81|
| cpp | 55.81 |
| rust | 52.98 |

> f"{name},{args.k},{pass_k},{num_problems},{min_completions},{max_completions}")
> humaneval-py-_opt_local_llm_models_huggingface.co_speechlessai_speechless_codellama_34b_v2.0-0.2-reworded,1,0.6754658385093167,161,20,20
> humaneval-java-_opt_local_llm_models_huggingface.co_speechlessai_speechless_codellama_34b_v2.0-0.2-reworded,1,0.5193037974683545,158,20,20
> humaneval-js-_opt_local_llm_models_huggingface.co_speechlessai_speechless_codellama_34b_v2.0-0.2-reworded,1,0.6481366459627329,161,20,20
> humaneval-cpp-_opt_local_llm_models_huggingface.co_speechlessai_speechless_codellama_34b_v2.0-0.2-reworded,1,0.5580745341614907,161,20,20
> humaneval-rs-_opt_local_llm_models_huggingface.co_speechlessai_speechless_codellama_34b_v2.0-0.2-reworded,1,0.5298076923076923,156,20,20

## speechless-codellama-34b-v1.9

### HumanEval

humaneval-python pass@1: 70.73

### MultiPL-E

| | |
| ------ | ------ |
|python | 56.02 |

> f"{name},{args.k},{pass_k},{num_problems},{min_completions},{max_completions}")
> humaneval-py-_opt_local_llm_models_huggingface.co_speechlessai_speechless_codellama_34b_v1.9-0.2-reworded,1,0.5602484472049689,161,20,20

## Data

## Evaluation

### HumanEval

Execute the HumanEval geenrate command on the GPU server where the model is located.

```bash
# make humaneval_gen
# call eval/humaneval_gen_vllm.py
bash ./eval/run_human_eval_gen.sh ${TEST_MODEL_PATH} ${HUMANEVAL_GEN_OUTPUT_FILE}

# make humaneval
python eval/run_humaneval.py \
    ${HUMANEVAL_GEN_OUTPUT_FILE} \
    --problem_file ${PWD}/eval/datasets/openai_humaneval/HumanEval.jsonl.gz
```

### MultiPL-E

```bash
MULTIPL_E_RESULTS_DIR=eval_results/multipl_e/${SERVED_MODEL_NAME}
SERVED_MODEL_NAME=$(shell basename ${TEST_MODEL_PATH})

make multipl_e_gen

# docker pull multipl-e-eval
make multipl_e_eval
# cd eval_results/multipl_e/${SERVED_MODEL_NAME} && bash ../../eval/run_multipl_e_eval.sh

make multipl_e_results
python ${PWD}/eval/MultiPL-E/pass_k.py -k 1 ${MULTIPL_E_RESULTS_DIR}/*
```
