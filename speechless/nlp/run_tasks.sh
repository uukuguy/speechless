#!/bin/bash
# This script demonstrates how to use the refactored run_classification_refactored.py
# to run training, evaluation, and prediction tasks separately.

# Set common parameters
MODEL_NAME="bert-base-uncased"
OUTPUT_DIR="./output"
MAX_SEQ_LENGTH=128
BATCH_SIZE=16

# Training only
echo "Running training only..."
python run_classification_refactored.py \
  --model_name_or_path $MODEL_NAME \
  --train_file ./data_splits/train_data_model_1.jsonl \
  --validation_file ./data_splits/val_data.jsonl \
  --text_column_names text \
  --output_dir $OUTPUT_DIR \
  --max_seq_length $MAX_SEQ_LENGTH \
  --per_device_train_batch_size $BATCH_SIZE \
  --per_device_eval_batch_size $BATCH_SIZE \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --do_train \
  --overwrite_output_dir

# Evaluation only
echo "Running evaluation only..."
python run_classification_refactored.py \
  --model_name_or_path $OUTPUT_DIR \
  --validation_file ./data_splits/val_data.jsonl \
  --text_column_names text \
  --output_dir $OUTPUT_DIR \
  --max_seq_length $MAX_SEQ_LENGTH \
  --per_device_eval_batch_size $BATCH_SIZE \
  --do_eval

# Prediction only
echo "Running prediction only..."
python run_classification_refactored.py \
  --model_name_or_path $OUTPUT_DIR \
  --test_file ./data/test.jsonl \
  --text_column_names text \
  --output_dir $OUTPUT_DIR \
  --max_seq_length $MAX_SEQ_LENGTH \
  --per_device_eval_batch_size $BATCH_SIZE \
  --do_predict

# Combined run (train, eval, and predict)
echo "Running combined (train, eval, and predict)..."
python run_classification_refactored.py \
  --model_name_or_path $MODEL_NAME \
  --train_file ./data_splits/train_data_model_1.jsonl \
  --validation_file ./data_splits/val_data.jsonl \
  --test_file ./data/test.jsonl \
  --text_column_names text \
  --output_dir $OUTPUT_DIR/combined \
  --max_seq_length $MAX_SEQ_LENGTH \
  --per_device_train_batch_size $BATCH_SIZE \
  --per_device_eval_batch_size $BATCH_SIZE \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --do_train \
  --do_eval \
  --do_predict \
  --overwrite_output_dir