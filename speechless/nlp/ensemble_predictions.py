#!/usr/bin/env python
# coding=utf-8
"""
Ensemble predictions from multiple models trained with cross-validation.
This script combines prediction results from multiple models to create a final prediction
with higher accuracy.
"""

import os
import argparse
import json
import numpy as np
from collections import Counter
from typing import List, Dict, Union, Tuple
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def load_predictions(prediction_files: List[str]) -> Tuple[List[Dict], List[List[Union[int, str]]]]:
    """
    Load prediction results from multiple files.
    
    Args:
        prediction_files: List of paths to prediction result files
        
    Returns:
        Tuple containing:
        - List of original examples (if available)
        - List of predictions from each model
    """
    all_predictions = []
    examples = []
    
    for i, file_path in enumerate(prediction_files):
        logger.info(f"Loading predictions from {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Prediction file not found: {file_path}")
        
        # Determine file type based on extension
        if file_path.endswith('.txt'):
            # Parse txt format (index\tprediction)
            predictions = []
            with open(file_path, 'r') as f:
                # Skip header line
                next(f)
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        # Try to convert prediction to int if possible
                        try:
                            pred = int(parts[1])
                        except ValueError:
                            # If not an int, keep as string (for multi-label or text predictions)
                            pred = parts[1]
                        predictions.append(pred)
            all_predictions.append(predictions)
            
        elif file_path.endswith('.json') or file_path.endswith('.jsonl'):
            # Parse JSON/JSONL format
            predictions = []
            loaded_examples = []
            
            with open(file_path, 'r') as f:
                for line in f:
                    example = json.loads(line.strip())
                    if i == 0:  # Only collect examples from the first file
                        loaded_examples.append(example)
                    
                    # Extract prediction based on expected format
                    if 'prediction' in example:
                        pred = example['prediction']
                    elif 'label' in example:
                        pred = example['label']
                    else:
                        raise ValueError(f"Could not find prediction in example: {example}")
                    
                    # Try to convert prediction to int if possible
                    try:
                        pred = int(pred)
                    except (ValueError, TypeError):
                        # If not an int, keep as is
                        pass
                    
                    predictions.append(pred)
            
            all_predictions.append(predictions)
            if i == 0 and loaded_examples:
                examples = loaded_examples
    
    # Verify all prediction lists have the same length
    lengths = [len(preds) for preds in all_predictions]
    if len(set(lengths)) > 1:
        raise ValueError(f"Prediction files have different numbers of predictions: {lengths}")
    
    logger.info(f"Loaded {len(all_predictions)} prediction sets with {lengths[0]} predictions each")
    return examples, all_predictions

def majority_vote(predictions: List[List[Union[int, str]]]) -> List[Union[int, str]]:
    """
    Combine predictions using majority voting.
    
    Args:
        predictions: List of prediction lists from different models
        
    Returns:
        List of ensemble predictions
    """
    ensemble_predictions = []
    
    # Transpose the predictions to get all model predictions for each example
    for i in range(len(predictions[0])):
        example_preds = [preds[i] for preds in predictions]
        # Count occurrences of each prediction
        counter = Counter(example_preds)
        # Get the most common prediction
        most_common = counter.most_common(1)[0][0]
        ensemble_predictions.append(most_common)
    
    return ensemble_predictions

def weighted_vote(predictions: List[List[Union[int, str]]], weights: List[float]) -> List[Union[int, str]]:
    """
    Combine predictions using weighted voting.
    
    Args:
        predictions: List of prediction lists from different models
        weights: List of weights for each model
        
    Returns:
        List of ensemble predictions
    """
    if len(predictions) != len(weights):
        raise ValueError(f"Number of prediction sets ({len(predictions)}) must match number of weights ({len(weights)})")
    
    ensemble_predictions = []
    
    # Transpose the predictions to get all model predictions for each example
    for i in range(len(predictions[0])):
        example_preds = [preds[i] for preds in predictions]
        # Count weighted occurrences of each prediction
        weighted_counter = {}
        for pred, weight in zip(example_preds, weights):
            weighted_counter[pred] = weighted_counter.get(pred, 0) + weight
        
        # Get the prediction with the highest weighted count
        max_pred = max(weighted_counter.items(), key=lambda x: x[1])[0]
        ensemble_predictions.append(max_pred)
    
    return ensemble_predictions

def average_probabilities(probability_files: List[str]) -> List[int]:
    """
    Combine predictions by averaging probabilities.
    
    Args:
        probability_files: List of paths to probability files (numpy arrays)
        
    Returns:
        List of ensemble predictions
    """
    all_probs = []
    
    for file_path in probability_files:
        logger.info(f"Loading probabilities from {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Probability file not found: {file_path}")
        
        # Load probabilities
        probs = np.load(file_path)
        all_probs.append(probs)
    
    # Average probabilities
    avg_probs = np.mean(all_probs, axis=0)
    
    # Get the class with the highest average probability for each example
    ensemble_predictions = np.argmax(avg_probs, axis=1).tolist()
    
    return ensemble_predictions

def save_predictions(output_file: str, predictions: List[Union[int, str]], examples: List[Dict] = None):
    """
    Save ensemble predictions to a file.
    
    Args:
        output_file: Path to output file
        predictions: List of ensemble predictions
        examples: Original examples (optional)
    """
    # Determine output format based on extension
    if output_file.endswith('.txt'):
        with open(output_file, 'w') as f:
            f.write("index\tprediction\n")
            for i, pred in enumerate(predictions):
                f.write(f"{i}\t{pred}\n")
    
    elif output_file.endswith('.json'):
        with open(output_file, 'w') as f:
            if examples:
                # If we have original examples, include them in the output
                for i, (example, pred) in enumerate(zip(examples, predictions)):
                    example['prediction'] = pred
                    f.write(json.dumps(example) + '\n')
            else:
                # Otherwise, just output the predictions
                for i, pred in enumerate(predictions):
                    f.write(json.dumps({"index": i, "prediction": pred}) + '\n')
    
    else:
        # Default to txt format
        with open(output_file, 'w') as f:
            f.write("index\tprediction\n")
            for i, pred in enumerate(predictions):
                f.write(f"{i}\t{pred}\n")
    
    logger.info(f"Saved {len(predictions)} ensemble predictions to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Ensemble predictions from multiple models")
    
    # Input arguments
    parser.add_argument(
        "--prediction_files",
        type=str,
        nargs="+",
        required=True,
        help="Paths to prediction files (txt, json, or jsonl format)",
    )
    parser.add_argument(
        "--probability_files",
        type=str,
        nargs="+",
        help="Paths to probability files (numpy arrays) for averaging probabilities",
    )
    
    # Ensemble method arguments
    parser.add_argument(
        "--method",
        type=str,
        default="majority",
        choices=["majority", "weighted", "average"],
        help="Ensemble method to use (majority: majority voting, weighted: weighted voting, average: average probabilities)",
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        help="Weights for each model (required for weighted voting)",
    )
    
    # Output arguments
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to output file for ensemble predictions",
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.method == "weighted" and not args.weights:
        raise ValueError("Weights must be provided for weighted voting")
    
    if args.method == "weighted" and len(args.weights) != len(args.prediction_files):
        raise ValueError(f"Number of weights ({len(args.weights)}) must match number of prediction files ({len(args.prediction_files)})")
    
    if args.method == "average" and not args.probability_files:
        raise ValueError("Probability files must be provided for averaging probabilities")
    
    # Load predictions
    examples, predictions = load_predictions(args.prediction_files)
    
    # Apply ensemble method
    if args.method == "majority":
        logger.info("Using majority voting ensemble method")
        ensemble_predictions = majority_vote(predictions)
    
    elif args.method == "weighted":
        logger.info(f"Using weighted voting ensemble method with weights: {args.weights}")
        ensemble_predictions = weighted_vote(predictions, args.weights)
    
    elif args.method == "average":
        logger.info("Using average probabilities ensemble method")
        ensemble_predictions = average_probabilities(args.probability_files)
    
    # Save ensemble predictions
    save_predictions(args.output_file, ensemble_predictions, examples)

if __name__ == "__main__":
    main()