# Selected MNIST ablation experiments

These are the highest- and lowest-accuracy variants from the fixed five-run
MNIST study of November 7, 2026. Each used all 60,000 training images for 10
epochs and was evaluated once on all 10,000 test images. Common settings:
seed 42, CPU with one thread, batch size 64, Adam at 0.001, cross-entropy loss,
raw float32 pixels 0–255, and no validation split.

| Model | Changed setting | Parameters | Test accuracy | Errors | Weighted F1 |
|---|---|---:|---:|---:|---:|
| Reference baseline | Two blocks, max pooling | 206,922 | 98.61% | 139 | 98.6090% |
| best_average_pool | Average pooling replaces max pooling | 206,922 | 99.01% | 99 | 99.0101% |
| worst_shallow | One convolutional block instead of two | 402,986 | 97.50% | 250 | 97.4983% |

Each variant has its own `Method_CNN_MNIST.py` with the modified default
configuration and a `results/` folder containing the original configuration,
metrics, epoch history, convergence plot, and trained `final.pt` checkpoint.
The shared loader and reproduction runner are in `code/`. The variants inherit
the instructor base class from `local_code/base_class/` in this repository.
The repository's original MNIST files are unchanged.

Only `pooling` changes in the average-pooling model; only `conv_channels`
changes in the shallow model. Removing the second convolution-and-pooling
block increases the flattened feature count from 1,568 to 3,136, so the shallow
model has more parameters despite having fewer convolutional layers.

## Run from the repository root

Use Python 3.10+ with NumPy, PyTorch, scikit-learn, and Matplotlib:

```bash
python -m pip install numpy torch scikit-learn matplotlib
```

Place the instructor's MNIST pickle at `stage_3_data/MNIST`, or supply
`--data-dir /path/to/folder-containing-MNIST`. The dataset is not uploaded.

Reproduce training (choose a new output folder each time):

```bash
python ablationTesting/mnist/code/run_ablation.py --variant best_average_pool --output ablationTesting/mnist/new_runs/average_pool_1
python ablationTesting/mnist/code/run_ablation.py --variant worst_shallow --output ablationTesting/mnist/new_runs/shallow_1
```

Evaluate the included checkpoints without training:

```bash
python ablationTesting/mnist/code/run_ablation.py --variant best_average_pool --evaluate-only
python ablationTesting/mnist/code/run_ablation.py --variant worst_shallow --evaluate-only
```

Evaluation prints accuracy, weighted precision/recall/F1, test loss, and the
confusion matrix. Add `--output <new-folder>` to also save evaluation metrics.
Training also saves configuration, epoch history, checkpoint, and convergence
plot. Included checkpoints are small (roughly 0.8 MB and 1.6 MB).

## Provenance and interpretation

`study_manifest.json` preserves the original experiment list, environment, and
source hashes. The selected model files were copied from the study source and
adapted only in default variant settings and repository import paths. Loader
paths and the reproduction runner were adapted for this repository layout.
`comparison.csv` includes just these two variants and the baseline reference.

Best/worst refer to observed accuracy within this five-run study, not a claim
that either architecture is universally best/worst. Each configuration used
one seed; small differences need repeated-seed experiments to assess reliability.
The test set was used to report and retrospectively select these two examples;
it should not be presented as an untouched holdout for further model tuning.
