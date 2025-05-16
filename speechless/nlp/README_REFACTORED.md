# Refactored Text Classification Script

This directory contains a refactored version of the text classification script that allows for separate execution of training, evaluation, and prediction tasks, as well as an ensemble prediction script for combining results from multiple models.

## Files

- `run_classification.py`: The original script that combines training, evaluation, and prediction.
- `run_classification_refactored.py`: The refactored script that separates the tasks.
- `run_tasks.sh`: A shell script demonstrating how to use the refactored script.
- `ensemble_predictions.py`: A script for combining predictions from multiple models.
- `run_ensemble.sh`: A shell script demonstrating how to use the ensemble prediction script.

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

## Handling Different Label Distributions

The refactored script includes several techniques to handle situations where the label distribution in the inference/prediction dataset may be very different from the training dataset. This is a common issue in machine learning known as "distribution shift" or "dataset shift".

### Available Techniques

1. **Temperature Scaling**: Adjust the softmax temperature to make the predictions more or less confident.
   - Higher temperature values make the model less confident (smoother probability distribution)
   - Lower temperature values make the model more confident (sharper probability distribution)

2. **Label Distribution Adjustment**: Explicitly adjust the predictions based on the expected label distribution in the test set.
   - This is useful when you know the approximate distribution of labels in the test set
   - The model's predictions are adjusted to match the expected distribution

3. **Decision Threshold Adjustment**: For binary classification, adjust the decision threshold based on the expected label distribution.
   - By default, the threshold is 0.5
   - Increasing the threshold makes the model more conservative about predicting the positive class
   - Decreasing the threshold makes the model more likely to predict the positive class

4. **Prediction Threshold Adjustment**: For multi-label classification, adjust the threshold for each label.
   - By default, the threshold is 0.0 (equivalent to using the sign of the logits)
   - Increasing the threshold makes the model more conservative about predicting each label

### Usage

To use these techniques, add the appropriate parameters when running prediction:

```bash
# Temperature scaling
python run_classification_refactored.py \
  --model_name_or_path ./output \
  --test_file ./data/test.jsonl \
  --text_column_names text \
  --output_dir ./output/temperature_scaling \
  --temperature 1.5 \
  --do_predict

# Label distribution adjustment
python run_classification_refactored.py \
  --model_name_or_path ./output \
  --test_file ./data/test.jsonl \
  --text_column_names text \
  --output_dir ./output/label_distribution \
  --expected_label_distribution '{"0": 0.7, "1": 0.3}' \
  --do_predict

# Decision threshold adjustment (for binary classification)
python run_classification_refactored.py \
  --model_name_or_path ./output \
  --test_file ./data/test.jsonl \
  --text_column_names text \
  --output_dir ./output/threshold_adjustment \
  --decision_threshold 0.7 \
  --do_predict

# Prediction threshold adjustment (for multi-label classification)
python run_classification_refactored.py \
  --model_name_or_path ./output \
  --test_file ./data/test.jsonl \
  --text_column_names text \
  --output_dir ./output/multi_label_threshold \
  --prediction_threshold 0.3 \
  --do_predict
```

These techniques can be combined with the ensemble prediction methods described earlier for even better results.

## Ensemble Predictions

The `ensemble_predictions.py` script allows you to combine predictions from multiple models trained with cross-validation to create a final prediction with higher accuracy.

### Ensemble Methods

The script supports three ensemble methods:

1. **Majority Voting**: Each model gets one vote, and the class with the most votes wins.
2. **Weighted Voting**: Similar to majority voting, but each model's vote is weighted based on its performance.
3. **Average Probabilities**: Averages the probability distributions from multiple models before making the final prediction.

### Usage

```bash
python ensemble_predictions.py \
  --prediction_files \
    ./output/model1/predict_results.txt \
    ./output/model2/predict_results.txt \
    ./output/model3/predict_results.txt \
  --method majority \
  --output_file ./output/ensemble/ensemble_majority.txt
```

For weighted voting:

```bash
python ensemble_predictions.py \
  --prediction_files \
    ./output/model1/predict_results.txt \
    ./output/model2/predict_results.txt \
    ./output/model3/predict_results.txt \
  --method weighted \
  --weights 1.0 1.2 0.8 \
  --output_file ./output/ensemble/ensemble_weighted.txt
```

You can also specify weights for individual labels to give more importance to certain classes:

```bash
python ensemble_predictions.py \
  --prediction_files \
    ./output/model1/predict_results.txt \
    ./output/model2/predict_results.txt \
    ./output/model3/predict_results.txt \
  --method majority \
  --label_weights '{"0": 1.0, "1": 2.0}' \
  --output_file ./output/ensemble/ensemble_label_weighted.txt
```

This is particularly useful for imbalanced datasets where some labels are more important than others. The label weights can be provided as a JSON string or a path to a JSON file.

For averaging probabilities:

```bash
python ensemble_predictions.py \
  --prediction_files \
    ./output/model1/predict_results.txt \
    ./output/model2/predict_results.txt \
    ./output/model3/predict_results.txt \
  --method average \
  --probability_files \
    ./output/model1/probabilities.npy \
    ./output/model2/probabilities.npy \
    ./output/model3/probabilities.npy \
  --output_file ./output/ensemble/ensemble_average.txt
```

### Using the Shell Script

For convenience, a shell script is provided that demonstrates how to train multiple models and create ensemble predictions:

```bash
./run_ensemble.sh
```

This script:
1. Trains three models using different training data splits
2. Makes predictions with each model
3. Creates ensemble predictions using majority voting and weighted voting