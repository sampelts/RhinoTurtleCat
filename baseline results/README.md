# Baseline model performance

Saved baseline run from October 19, 2026 (before the one-hidden-layer ablation).
Source code commit: `18135b1200374d79a220b9f172368c8b2c0a825a`.

## Model and training settings

- Architecture: 784 → 256 → 128 → 10, with ReLU after each hidden layer.
- Optimizer: Adam; learning rate: 0.001.
- Training: 300 epochs using the entire training set as one batch.
- NumPy and PyTorch seed: 2.
- Input: original CSV pixel values, without normalization.
- Split: instructor-provided train/test files; no validation split.

## Final test performance

Calculated from the saved baseline predictions for 10,000 test examples.
These values match the completed run's console output; training was not rerun.
Accuracy, precision, recall, and F1 are shown on a 0–1 scale.
Macro metrics average over digits 0 through 9 with zero_division=0.

| Metric | Value |
| --- | ---: |
| Accuracy | 0.9659000000 |
| Precision (macro) | 0.9657306297 |
| Recall (macro) | 0.9655256522 |
| F1 (macro) | 0.9655847541 |

Test loss was not recorded by this version of the code.

## Last recorded training performance

| Metric | Value |
| --- | ---: |
| Epoch (zero-based) | 299 |
| Accuracy | 0.9982 |
| Precision (macro) | 0.9982116568450463 |
| Recall (macro) | 0.9981944802719667 |
| F1 (macro) | 0.9982029523351397 |
| Cross-entropy loss | 0.01414792612195015 |

Training metrics use the forward pass before that epoch's optimizer update.
Final test evaluation uses the weights after all 300 updates.

## Saved history

`history_MLP.csv` is an unchanged copy of `result/stage_2_result/history_MLP.csv`.
It contains 31 recorded training rows: epochs 0, 10, 20, …, 290, and 299.
Its columns are Epoch, Accuracy, Precision_Macro, Recall_Macro, F1_Macro, and Loss.

This folder is a snapshot. Running the model again writes to `result/`, leaving
these archived baseline results unchanged. The raw prediction pickle and dataset
are not included in this folder.
