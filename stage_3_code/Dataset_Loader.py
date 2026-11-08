
"""Load the instructor's CIFAR pickle as N x 3 x 32 x 32 image arrays."""

import pickle
from pathlib import Path
import numpy as np

from local_code.base_class.dataset import dataset

# Location of the instructor-provided datasets
DATA_DIRECTORY = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "stage_3_data"
)


class Dataset_Loader_CIFAR(dataset):

    def __init__(self, split="train", data_dir=None):
        super().__init__("CIFAR", "Colored object images")

        if split not in {"train", "test"}:
            raise ValueError("split must be train or test")

        self.split = split

        self.dataset_source_folder_path = (
            Path(data_dir) if data_dir is not None
            else DATA_DIRECTORY
        )

        self.dataset_source_file_name = "CIFAR"

    def load(self):
        print("loading CIFAR data...")

        path = (
            Path(self.dataset_source_folder_path)
            / self.dataset_source_file_name
        )

        # Load the instructor's pickle dataset
        with path.open("rb") as file:
            data = pickle.load(file)

        # Select training or testing data
        examples = data[self.split]

        # Convert images to NumPy arrays
        images = np.asarray(
            [instance["image"] for instance in examples],
            dtype=np.float32
        )

        # Convert NHWC to NCHW for PyTorch
        # (N, 32, 32, 3) -> (N, 3, 32, 32)
        if images.ndim != 4:
            raise ValueError("Expected 4D CIFAR image data")

        if images.shape[1:] == (32, 32, 3):
            images = images.transpose(0, 3, 1, 2)
        elif images.shape[1:] != (3, 32, 32):
            raise ValueError(
                f"Unexpected CIFAR image shape: {images.shape}"
            )

        # Normalize pixels to 0-1
        # Assumes original pixel values are 0-255
        images = images / 255.0

        # Convert labels to integers
        labels = np.asarray(
            [instance["label"] for instance in examples],
            dtype=np.int64
        )

        self.data = {
            "X": images,
            "y": labels,
            "summary": {
                "dataset": "CIFAR",
                "split": self.split,
                "samples": len(labels),
                "image_shape": [3, 32, 32],
                "class_counts": np.bincount(
                    labels, minlength=10
                ).tolist(),
                "scaling": "float32 pixels in 0..1"
            }
        }

        return self.data
