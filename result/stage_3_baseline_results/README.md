# Stage 3 Baseline Results

This folder exists only on the `Stage-3` branch. It stores report-ready,
lightweight output files for the baseline Stage 3 MNIST CNN, separate from the
source code and without a large model checkpoint.

## MNIST baseline

`mnist_baseline/` contains the baseline CNN run trained for 10 epochs on all
60,000 supplied MNIST training images. It was evaluated once on the separate
10,000-image test set and achieved 98.61% accuracy (9,861 correct images).

- `config.json`: hyperparameters and run configuration
- `history.csv`: training loss and accuracy for every epoch
- `metrics.json`: final test metrics and confusion matrix
- `learning_curve.png`: loss and accuracy curves
- `training_convergence.png`: epoch-versus-training-loss plot
