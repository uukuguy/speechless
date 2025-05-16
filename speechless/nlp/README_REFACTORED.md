# Refactored Text Classification Script

This directory contains a refactored version of the text classification script that allows for separate execution of training, evaluation, and prediction tasks.

## Files

- `run_classification.py`: The original script that combines training, evaluation, and prediction.
- `run_classification_refactored.py`: The refactored script that separates the tasks.
- `run_tasks.sh`: A shell script demonstrating how to use the refactored script.

## Key Improvements

1. **Separation of Tasks**: The refactored script allows you to run training, evaluation, and prediction tasks separately or in combination.
2. **Modular Design**: The code is organized into separate functions for each task, making it easier to understand and maintain.
3. **Reduced Parameter Requirements**: When running prediction only, you don't need to provide training-specific parameters.
4. **Better Error Handling**: The script includes improved error handling and logging.

## Usage

### Command Line Arguments

In addition to the original arguments, the refactored script includes a new `TaskArguments` class with the following options:

- `--do_train`: Whether to run training.
- `--do_eval`: Whether to run evaluation on the validation set.
- `--do_predict`: Whether to run predictions on the test set.

### Examples

#### Training Only

```bash
python run_classification_refactored.py \
  --model_name_or_path bert-base-uncased \
  --train_file ./data_splits/train_data_model_1.jsonl \
  --validation_file ./data_splits/val_data.jsonl \
  --text_column_names text \
  --output_dir ./output \
  --max_seq_length 128 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --do_train \
  --overwrite_output_dir
```

#### Evaluation Only

```bash
python run_classification_refactored.py \
  --model_name_or_path ./output \
  --validation_file ./data_splits/val_data.jsonl \
  --text_column_names text \
  --output_dir ./output \
  --max_seq_length 128 \
  --per_device_eval_batch_size 16 \
  --do_eval
```

#### Prediction Only

```bash
python run_classification_refactored.py \
  --model_name_or_path ./output \
  --test_file ./data/test.jsonl \
  --text_column_names text \
  --output_dir ./output \
  --max_seq_length 128 \
  --per_device_eval_batch_size 16 \
  --do_predict
```

#### Combined (Train, Eval, and Predict)

```bash
python run_classification_refactored.py \
  --model_name_or_path bert-base-uncased \
  --train_file ./data_splits/train_data_model_1.jsonl \
  --validation_file ./data_splits/val_data.jsonl \
  --test_file ./data/test.jsonl \
  --text_column_names text \
  --output_dir ./output/combined \
  --max_seq_length 128 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --do_train \
  --do_eval \
  --do_predict \
  --overwrite_output_dir
```

### Using the Shell Script

For convenience, a shell script is provided that demonstrates how to use the refactored script:

```bash
./run_tasks.sh
```

This script runs four examples:
1. Training only
2. Evaluation only
3. Prediction only
4. Combined (train, eval, and predict)

## Implementation Details

The refactored script is organized into the following main functions:

1. `load_datasets`: Loads datasets from either a dataset name or local files.
2. `preprocess_datasets`: Preprocesses datasets by tokenizing and converting labels to IDs.
3. `load_model_and_tokenizer`: Loads the model and tokenizer.
4. `train_model`: Trains the model.
5. `evaluate_model`: Evaluates the model.
6. `predict_with_model`: Makes predictions with the model.
7. `main`: The main function that orchestrates the entire process.

Each function is designed to be independent and reusable, making the code more maintainable and easier to understand.