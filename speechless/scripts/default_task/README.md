# Speechless Scripts

## How to

### Fine-tune

```bash
# Step 1: 
cp ../../scripts/default_task/* .

# Step 2: Edit task.env

# Step 3: 
./run_finetune.sh
```

### Merge LoRA models

```bash
# Use the TEST_MODEL_PATH in task.env
./merge_peft_adapters.sh
```

### Run API Server

```bash
../../scripts/run_api_server.sh <model_name_or_path>
```

### HumanEval

```bash
./run_humeneval_gen.sh
./run_humeeval.sh
```

### SQLEval

Step 1: Start the API server by execute `run_api_server.sh`.

Step 2: In Client, run `make codellama` in speechless/eval/sql-eval .

### MultiPL-E

### LMEval

