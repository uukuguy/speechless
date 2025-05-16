#!/bin/bash
# This script demonstrates how to use the ensemble_predictions.py script
# to combine predictions from multiple cross-validation models.

# Set common parameters
OUTPUT_DIR="./output"
TEST_FILE="./data/test.jsonl"
MODEL_NAME="bert-base-uncased"
MAX_SEQ_LENGTH=128
BATCH_SIZE=16

# Create output directories for each model
mkdir -p $OUTPUT_DIR/model1
mkdir -p $OUTPUT_DIR/model2
mkdir -p $OUTPUT_DIR/model3
mkdir -p $OUTPUT_DIR/ensemble

# Train and predict with Model 1
echo "Training and predicting with Model 1..."
python run_classification_refactored.py \
  --model_name_or_path $MODEL_NAME \
  --train_file ./data_splits/train_data_model_1.jsonl \
  --validation_file ./data_splits/val_data.jsonl \
  --test_file $TEST_FILE \
  --text_column_names text \
  --output_dir $OUTPUT_DIR/model1 \
  --max_seq_length $MAX_SEQ_LENGTH \
  --per_device_train_batch_size $BATCH_SIZE \
  --per_device_eval_batch_size $BATCH_SIZE \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --do_train \
  --do_predict \
  --overwrite_output_dir

# Train and predict with Model 2
echo "Training and predicting with Model 2..."
python run_classification_refactored.py \
  --model_name_or_path $MODEL_NAME \
  --train_file ./data_splits/train_data_model_2.jsonl \
  --validation_file ./data_splits/val_data.jsonl \
  --test_file $TEST_FILE \
  --text_column_names text \
  --output_dir $OUTPUT_DIR/model2 \
  --max_seq_length $MAX_SEQ_LENGTH \
  --per_device_train_batch_size $BATCH_SIZE \
  --per_device_eval_batch_size $BATCH_SIZE \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --do_train \
  --do_predict \
  --overwrite_output_dir

# Train and predict with Model 3
echo "Training and predicting with Model 3..."
python run_classification_refactored.py \
  --model_name_or_path $MODEL_NAME \
  --train_file ./data_splits/train_data_model_3.jsonl \
  --validation_file ./data_splits/val_data.jsonl \
  --test_file $TEST_FILE \
  --text_column_names text \
  --output_dir $OUTPUT_DIR/model3 \
  --max_seq_length $MAX_SEQ_LENGTH \
  --per_device_train_batch_size $BATCH_SIZE \
  --per_device_eval_batch_size $BATCH_SIZE \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --do_train \
  --do_predict \
  --overwrite_output_dir

# Ensemble predictions using majority voting
echo "Creating ensemble prediction with majority voting..."
python ensemble_predictions.py \
  --prediction_files \
    $OUTPUT_DIR/model1/predict_results.txt \
    $OUTPUT_DIR/model2/predict_results.txt \
    $OUTPUT_DIR/model3/predict_results.txt \
  --method majority \
  --output_file $OUTPUT_DIR/ensemble/ensemble_majority.txt

# Ensemble predictions using weighted voting
# Weights are based on validation performance (example values)
echo "Creating ensemble prediction with weighted voting..."
python ensemble_predictions.py \
  --prediction_files \
    $OUTPUT_DIR/model1/predict_results.txt \
    $OUTPUT_DIR/model2/predict_results.txt \
    $OUTPUT_DIR/model3/predict_results.txt \
  --method weighted \
  --weights 1.0 1.2 0.8 \
  --output_file $OUTPUT_DIR/ensemble/ensemble_weighted.txt

echo "Ensemble predictions completed. Results saved to $OUTPUT_DIR/ensemble/"