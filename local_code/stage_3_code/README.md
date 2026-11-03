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

## ORL face recognition (Samantha's implementation)

The ORL implementation was integrated from commit `9fa2839` on
`Stage-3-Samantha`. The MNIST loader, model, and runner remain Andrew's versions.

ORL uses `Dataset_Loader_Image.py`, `Method_CNN_ORL.py`, and its shared training
helper `Method_CNN_Base.py`, along with `Evaluate_Accuracy.py`,
`History_Saver.py`, `Result_Saver.py`, and `Setting_Train_Test_PreSplit.py`.

Place the instructor's trusted ORL pickle at `data/stage_3_data/ORL` relative
to the repository root. Install the dependencies listed above, then run:

```bash
python script/stage_3_script/script_orl.py
```

The loader uses one grayscale channel, converts labels 1–40 to 0–39, and
standardizes images using statistics computed from training images only.
This preprocessing is specific to Samantha's ORL experiment; MNIST still uses
raw pixel values. The four convolutional blocks use 16, 32, 64, and 64 filters,
batch normalization, ReLU, and max pooling. A 128-unit dense layer and dropout
precede the 40-class output. Defaults are 70 epochs, batch size 32, Adam,
learning rate 0.0005, and cross-entropy loss.

The runner prints final accuracy and macro precision, recall, and F1. It saves
`history_orl.csv` and a pickle containing predictions and true labels under
`result/stage_3_baseline_results/orl_baseline/`. Re-running overwrites these
ORL output files. No ORL baseline scores are claimed by the integration check.

Integration fixes make runner paths independent of the working directory,
evaluate all observed ORL classes, store scalar training accuracy, validate
the selected channel index, and open history files for reading when loading.
