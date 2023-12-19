from pcap_data_preprocess import build_pcap_data
import random
import json
import os


MAX_SAMPLING_NUMBER = 5000  # 5000
TRAINING_SAMPLE_RATIO = 1.0


def split_dataset(build_data, sampling=True):
    random.shuffle(build_data)
    if sampling is True:
        train_nb = int(min(MAX_SAMPLING_NUMBER, len(build_data)) * TRAINING_SAMPLE_RATIO)
    else:
        train_nb = int(len(build_data) * TRAINING_SAMPLE_RATIO)
    train_data = build_data[:train_nb]

    return train_data


def write_dataset(dataset, output_path):
    random.shuffle(dataset)
    with open(output_path, "w", encoding="utf-8") as fin:
        for data in dataset:
            json.dump(data, fin, ensure_ascii=False)
            fin.write("\n")


def write_labels(labels, output_path):
    label_dict = {}
    for i, label in enumerate(labels):
        label_dict[label] = i
    with open(output_path, "w", encoding="utf-8") as fin:
        json.dump(label_dict, fin, indent=4, separators=(',', ': '), ensure_ascii=False)


def build_dataset(path, file, is_train):
    build_data = []
    files_path = os.path.join(path, file)
    pcaps = os.listdir(files_path)
    for pcap in pcaps:
        pcap_data = build_pcap_data(os.path.join(files_path, pcap))
        build_data.extend(pcap_data)

    if is_train is True:
        train_data = split_dataset(build_data, sampling=True)
    else:
        train_data = build_data

    return train_data


def save_dataset(args, train_dataset):
    write_dataset(train_dataset, os.path.join(args.output_path, args.output_name + "_train.json"))


def build_td_text_dataset(traffic_data, label=None, task_name=None):
    """Building the text datasets of traffic detection task"""

    if task_name == "EMD":
        instruction = "以下为一段网络流量数据，请执行恶意流量识别任务："

        output = "这段流量数据的类别为" + label + "。"

    elif task_name == "BND":
        instruction = "以下为一段网络流量数据，请执行僵尸网络检测任务："

        output = "这段流量数据的类别为" + label + "。"

    elif task_name == "EVD":
        instruction = "以下为一段网络流量数据，请执行隧道行为识别任务："

        output = "这段流量数据的类别为" + label + "。"

    dataset = []
    for data in traffic_data:
        dataset.append(
            {
                "instruction": instruction + data,
                "output": output
            }
        )

    return dataset
