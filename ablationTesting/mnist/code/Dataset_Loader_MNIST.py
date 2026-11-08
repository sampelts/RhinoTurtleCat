"""Load the instructor's MNIST pickle as N x 1 x 28 x 28 image tensors."""

import pickle
from pathlib import Path

import numpy as np

from local_code.base_class.dataset import dataset

DATA_DIRECTORY = Path(__file__).resolve().parents[3] / "stage_3_data"


class Dataset_Loader_MNIST(dataset):
    """Load the instructor's MNIST train or test list for the digit CNN."""

    def __init__(self, *, data_dir=None):
        super().__init__("MNIST", "Handwritten digit images")
        self.dataset_source_folder_path = Path(data_dir) if data_dir is not None else DATA_DIRECTORY
        self.dataset_source_file_name = "MNIST"

    def load(self, split="train"):
        """Follow the instructor's pickle format and return channel-first MNIST arrays."""
        if split not in {"train", "test"}:
            raise ValueError("split must be train or test")
        path = Path(self.dataset_source_folder_path) / self.dataset_source_file_name
        with path.open("rb") as stream:
            data = pickle.load(stream)
        examples = data[split]
        images = np.asarray([instance["image"] for instance in examples], dtype=np.float32)[:, None, :, :]
        labels = np.asarray([instance["label"] for instance in examples], dtype=np.int64)
        self.data = {
            "X": images,
            "y": labels,
            "summary": {
                "dataset": "MNIST",
                "split": split,
                "samples": len(labels),
                "image_shape": [1, 28, 28],
                "class_counts": np.bincount(labels, minlength=10).tolist(),
                "scaling": "raw float32 pixels in 0..255",
            },
        }
        return self.data
