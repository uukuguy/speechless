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

# Prediction only (standard)
echo "Running standard prediction..."
python run_classification_refactored.py \
  --model_name_or_path $OUTPUT_DIR \
  --test_file ./data/test.jsonl \
  --text_column_names text \
  --output_dir $OUTPUT_DIR/standard_prediction \
  --max_seq_length $MAX_SEQ_LENGTH \
  --per_device_eval_batch_size $BATCH_SIZE \
  --do_predict

# Prediction with temperature scaling
echo "Running prediction with temperature scaling..."
python run_classification_refactored.py \
  --model_name_or_path $OUTPUT_DIR \
  --test_file ./data/test.jsonl \
  --text_column_names text \
  --output_dir $OUTPUT_DIR/temperature_scaling \
  --max_seq_length $MAX_SEQ_LENGTH \
  --per_device_eval_batch_size $BATCH_SIZE \
  --temperature 1.5 \
  --do_predict

# Prediction with label distribution adjustment
echo "Running prediction with label distribution adjustment..."
python run_classification_refactored.py \
  --model_name_or_path $OUTPUT_DIR \
  --test_file ./data/test.jsonl \
  --text_column_names text \
  --output_dir $OUTPUT_DIR/label_distribution \
  --max_seq_length $MAX_SEQ_LENGTH \
  --per_device_eval_batch_size $BATCH_SIZE \
  --expected_label_distribution '{"0": 0.7, "1": 0.3}' \
  --do_predict

# Prediction with decision threshold adjustment (for binary classification)
echo "Running prediction with decision threshold adjustment..."
python run_classification_refactored.py \
  --model_name_or_path $OUTPUT_DIR \
  --test_file ./data/test.jsonl \
  --text_column_names text \
  --output_dir $OUTPUT_DIR/threshold_adjustment \
  --max_seq_length $MAX_SEQ_LENGTH \
  --per_device_eval_batch_size $BATCH_SIZE \
  --decision_threshold 0.7 \
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