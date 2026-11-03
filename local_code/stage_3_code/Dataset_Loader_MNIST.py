"""Load the instructor's MNIST pickle as N x 1 x 28 x 28 image tensors."""

# Read the instructor-provided binary pickle dataset.
import pickle
# Build platform-independent paths to the data file.
from pathlib import Path

# Store image pixels and labels in efficient numerical arrays.
import numpy as np

# Inherit from the instructor's required dataset base class.
from local_code.base_class.dataset import dataset

# Point to the expected repository-level location of the instructor data.
DATA_DIRECTORY = Path(__file__).resolve().parents[2] / "stage_3_data"


# Define the MNIST-specific dataset loader used by the CNN script.
class Dataset_Loader_MNIST(dataset):
    """Load the instructor's MNIST train or test list for the digit CNN."""

    # Initialize the loader and optionally allow a different data directory.
    def __init__(self, *, data_dir=None):
        # Store the dataset name and description through the base class.
        super().__init__("MNIST", "Handwritten digit images")
        # Use a caller-provided directory, or the standard Stage 3 data folder.
        self.dataset_source_folder_path = Path(data_dir) if data_dir is not None else DATA_DIRECTORY
        # The supplied MNIST pickle filename has no extension.
        self.dataset_source_file_name = "MNIST"

    # Load either the supplied training or testing split.
    def load(self, split="train"):
        """Follow the instructor's pickle format and return channel-first MNIST arrays."""
        # Reject an invalid split name before trying to read the file.
        if split not in {"train", "test"}:
            raise ValueError("split must be train or test")
        # Join the data directory and the MNIST pickle filename.
        path = Path(self.dataset_source_folder_path) / self.dataset_source_file_name
        # Open the pickle safely in binary-read mode.
        with path.open("rb") as stream:
            # Deserialize the dictionary containing train and test examples.
            data = pickle.load(stream)
        # Select the requested list of image-and-label dictionaries.
        examples = data[split]
        # Create float32 images and add the one grayscale-channel dimension.
        images = np.asarray([instance["image"] for instance in examples], dtype=np.float32)[:, None, :, :]
        # Create integer class labels that PyTorch cross-entropy loss expects.
        labels = np.asarray([instance["label"] for instance in examples], dtype=np.int64)
        # Save tensors-ready arrays and useful dataset information on this object.
        self.data = {
            # X contains N x 1 x 28 x 28 MNIST image arrays.
            "X": images,
            # y contains the corresponding digit labels from 0 through 9.
            "y": labels,
            # summary records key facts for inspection or reporting.
            "summary": {
                # Record the dataset identity.
                "dataset": "MNIST",
                # Record whether this is the train or test split.
                "split": split,
                # Record how many examples were loaded.
                "samples": len(labels),
                # Record the channel-first shape of one example.
                "image_shape": [1, 28, 28],
                # Count examples from each digit class, including empty classes.
                "class_counts": np.bincount(labels, minlength=10).tolist(),
                # State that no division by 255 is applied in this experiment.
                "scaling": "raw float32 pixels in 0..255",
            },
        }
        # Return the loaded data dictionary to the training or test script.
        return self.data
