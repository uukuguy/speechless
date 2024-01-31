# NUGGETS

> speechless.synthetic_data.nuggets

[NUGGETS](https://github.com/pldlgb/nuggets) - a novel and efficient methodology that employs one shot learning to select high-quality instruction data from expansive datasets.

> make do_test_nuggets

```bash
MODEL_PATH=/opt/local/llm_models/huggingface.co/mistralai/Mistral-7B-v0.1 \
PROMPT_PATH=/opt/local/datasets/alpaca_gpt4/alpaca_gpt4_data.json \
TEST_PATH=/opt/local/datasets/alpaca_gpt4/alpaca_gpt4_kmeans_100.json \
BATCH_SIZE=0 \
python -m speechless.synthetic_data.nuggets \
    --dataset alpaca \
    --exemplar_method stratified --num_k_shots 1 \
    --model_path ${MODEL_PATH} \
    --prompt_path ${PROMPT_PATH} \
    --test_path ${TEST_PATH} \
    --save_path "save/alpaca_gpt4/score" \
    --kv_iter 1 \
    --step_size 0.01 --momentum 0.9 \
    --batch_size ${BATCH_SIZE} \
    --start $start \ 
    --pace $pace
```
