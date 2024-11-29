import json
from pathlib import Path


class Dataset:
    """
    Light-weight wrapper to hold data a jsonl file for use in training, validation, and testing with mlx

    Extended to support Hugging Face datasets
    """

    def __init__(self, path: Path):
        if not path.exists():
            self._data = None
        else:
            with open(path, "r") as f:
                self._data = [json.loads(line) for line in f]

    def __getitem__(self, idx: int):
        return self._data[idx]

    def __len__(self):
        return len(self._data)
