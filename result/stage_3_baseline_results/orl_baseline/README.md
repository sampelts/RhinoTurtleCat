# Sammy's ORL baseline results

The original `history_orl.csv` and `CNN_ORLprediction_result_None` were supplied
by Sammy. The prediction file is a Python pickle containing PyTorch tensors
`pred_y` and `true_y`; it is prediction output, not a trained model checkpoint.
The extensionless filename is valid. Load pickle files only from trusted sources.

## Verified test metrics

The saved true labels match the instructor ORL test split in order after
subtracting 1 to obtain labels 0–39. There are 40 test examples, one per person.
All 40 predictions are correct. Accuracy, macro precision/recall/F1, and weighted
precision/recall/F1 are all 1.0 (100%). These results describe this small supplied
test split; the artifacts alone cannot establish how training or model selection
was conducted, or independently reproduce the trained model's predictions.

`metrics.json` contains metrics recomputed over all 40 classes and a confusion
matrix. `predictions.csv` provides the same labels in a portable text format.
Cross-entropy test loss cannot be recovered without saved class scores/logits.

## Training history and convergence

The history contains 70 epochs numbered 0–69. The final training loss is
0.0508006941 and the final recorded training accuracy is 0.9944444444 (99.44%).
The original Accuracy column embeds a metrics dictionary. Its precision, recall,
and F1 values should not be used as final ORL test results: the earlier evaluator
hard-coded labels 0–9. `history_clean.csv` extracts only training loss and overall
accuracy and numbers the epochs 1–70 for plotting. `training_convergence.png`
plots the recorded training loss; it is not a new training run.

## Regenerate derived files

From the repository root, after installing the MNIST/ORL dependencies:

```bash
python script/stage_3_script/summarize_orl_results.py --predictions result/stage_3_baseline_results/orl_baseline/CNN_ORLprediction_result_None --history result/stage_3_baseline_results/orl_baseline/history_orl.csv --output result/stage_3_baseline_results/orl_summary
```

For full experiment reproduction, also retain the actual run configuration
(seed, epochs, optimizer, learning rate, architecture and preprocessing) and
optionally a trained checkpoint. The supplied two files do not record a complete
run configuration, so no configuration file has been invented for this run.
