# Stage 3 MNIST CNN

This folder contains the MNIST implementation for Stage 3: a data loader,
a convolutional neural network, and a script that trains and evaluates it.
The classes inherit from the supplied `local_code/base_class/dataset.py` and
`local_code/base_class/method.py` templates.

## Required data

The instructor-provided MNIST pickle file is not included in this repository.
Place it at this exact location after cloning the repository:

```text
stage_3_data/MNIST
```

The pickle must contain the supplied `train` and `test` splits.  The loader
keeps pixels as raw `float32` values in the 0–255 range, adds a grayscale
channel dimension, and uses labels 0–9.

## Install dependencies

Use Python 3.10 or newer. From the repository root, create and activate a
virtual environment if desired, then install:

```bash
python -m pip install numpy torch scikit-learn matplotlib
```

## Run MNIST

Run this command from the repository root:

```bash
python script/stage_3_script/script_mnist.py --run-name mnist_final --epochs 10
```

`--run-name` must be new for every run. The default configuration uses 64-image
mini-batches, Adam with learning rate 0.001, 10 epochs, two convolutional
blocks (16 and 32 filters), ReLU, 2 × 2 max pooling, and a 128-unit dense layer.
All 60,000 supplied training images are used to train the model; the supplied
10,000-image test split is evaluated once after training.

## Outputs

Each run is saved to `result/stage_3_result/mnist/<run-name>/`:

- `metrics.json`: final test loss, accuracy, weighted precision, recall, F1,
  and confusion matrix.
- `history.csv`: training loss and accuracy after each epoch.
- `learning_curve.png`: combined loss and accuracy learning curves.
- `training_convergence.png`: report-ready epoch-versus-training-loss plot.
- `config.json`: run configuration.
- `final.pt`: trained checkpoint (generated locally; not tracked in Git).

The reported reference run (`mnist_simple_full_v1`) achieved 98.61% test
accuracy on the supplied 10,000-image test set.
