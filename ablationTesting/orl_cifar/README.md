# ORL and CIFAR configuration experiments (Stage 3, step 3-5)

Companion to `ablationTesting/mnist/`. 12 configurations per dataset, each changing one setting of the
baseline CNN: depth, kernel size, padding, stride, pooling, hidden size, batch norm, dropout, loss, optimizer.

## Setup
- Model code: Sai's shared `Method_CNN` (branch `sai-changes`), baselines `Method_CNN_ORL` and `Method_CNN_CIFAR`.
- 10% of the supplied training split is a stratified validation split
  (ORL 320 train / 40 val, 1 image per person; CIFAR 45,000 train / 5,000 val).
  Configurations are compared on validation; the supplied test split is evaluated once per run.
- ORL: 50 epochs, seeds 0, 1, 2 (test split has 40 images, so 1 image = 2.5 points).
- CIFAR: 20 epochs, seed 0. Adam lr 1e-3, standardized inputs. Colab T4 GPU, about 13 minutes in total.

## Results (test accuracy %, change vs baseline)
See `results/ablation_table.md` (report 3.7), `results/baseline_metrics.md` (report 3.6),
`results/convergence.png` (report 3.5). Every run's metrics are in `results/runs/<DATASET>/<config>/seed<k>/metrics.json`.

## Reproduce
Open `code/stage3_ablation_ORL_CIFAR_colab.ipynb` in Colab and run all cells. It clones `sai-changes`,
writes the four files in `code/` into it, downloads the course data, runs every configuration and rebuilds the tables.
