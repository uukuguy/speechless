import os
import random


detection_datasets = [
    "botnet",
    "vpn",
    "malware"
]

output_path = "utils/build_datasets/"
output_name = "training.json"

if __name__ == "__main__":
    print("Merge training datasets: " + os.path.join(output_path, output_name))

    datasets = []

    for dataset in detection_datasets:

        data_dir = os.path.join("utils/build_datasets", dataset)
        files = os.listdir(data_dir)

        for file in files:
            # if "train" in file and "flow" in file:
            with open(os.path.join(data_dir, file), "r", encoding="utf-8") as fin:
                datasets.extend(fin.readlines())

    random.shuffle(datasets)

    with open(os.path.join(output_path, output_name), "w", encoding="utf-8") as fin:
        fin.writelines(datasets)
