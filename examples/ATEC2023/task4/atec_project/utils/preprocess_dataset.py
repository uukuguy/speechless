from preprocess_utils import (
    build_td_text_dataset,
    write_labels,
    build_dataset,
    save_dataset
)
from tqdm import tqdm
import argparse
import random
import os


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="raw dataset path", required=True)
    parser.add_argument("--dataset_name", type=str, help="dataset name", required=True)
    parser.add_argument("--output_path", type=str, help="output dataset path", required=True)
    parser.add_argument("--output_name", type=str, help="output dataset name", required=True)

    args = parser.parse_args()
    return args


def traffic_detection_preprocess(args, detection_task):
    """Dataset preprocessing for the traffic detection (TD) task"""
    train_dataset = []
    labels = []

    files = os.listdir(args.input)
    labels.extend(files)

    for file in tqdm(files):
        train_data = build_dataset(args.input, file, is_train=True)

        train_text_data = build_td_text_dataset(train_data, label=file, task_name=detection_task)

        train_dataset.extend(train_text_data)

    save_dataset(args, train_dataset)

    # write_labels(labels, os.path.join(args.output_path, args.output_name + "_label.json"))


def main():
    args = get_args()

    if args.dataset_name == "malware":
        traffic_detection_preprocess(args, detection_task="EMD")
    elif args.dataset_name == "botnet":
        traffic_detection_preprocess(args, detection_task="BND")
    elif args.dataset_name == "vpn":
        traffic_detection_preprocess(args, detection_task="EVD")


if __name__ == "__main__":
    main()
