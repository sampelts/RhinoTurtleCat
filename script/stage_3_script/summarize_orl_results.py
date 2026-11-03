"""Convert a trusted ORL prediction pickle and training history into report outputs."""

import argparse
import ast
import csv
import json
from pathlib import Path
import pickle
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Only load trusted pickle files, such as the supplied team experiment output.
    with args.predictions.open("rb") as stream:
        saved = pickle.load(stream)
    truth = np.asarray(saved["true_y"], dtype=np.int64)
    predicted = np.asarray(saved["pred_y"], dtype=np.int64)
    if truth.shape != (40,) or predicted.shape != truth.shape:
        raise ValueError("Expected 40 ORL test labels and matching predictions")
    if set(truth.tolist()) != set(range(40)) or not np.isin(predicted, range(40)).all():
        raise ValueError("Expected ORL labels 0–39, one true example per person")
    with args.history.open(newline="") as stream:
        original = list(csv.DictReader(stream))
    history = []
    for row in original:
        # Sammy's original Accuracy column stores a metrics dictionary as text.
        accuracy = ast.literal_eval(row["Accuracy"])
        history.append({"epoch": int(row["Epoch"]) + 1,
                        "train_loss": float(row["Loss"]),
                        "train_accuracy": float(accuracy["accuracy"] if isinstance(accuracy, dict) else accuracy)})
    metrics = {"samples": len(truth), "classes": 40,
               "correct": int((truth == predicted).sum()),
               "accuracy": float(accuracy_score(truth, predicted)),
               "test_loss": None,
               "test_loss_note": "Class predictions alone do not retain logits needed for cross-entropy loss."}
    for average in ("macro", "weighted"):
        p, r, f1, _ = precision_recall_fscore_support(
            truth, predicted, labels=list(range(40)), average=average, zero_division=0)
        metrics.update({f"{average}_precision": float(p), f"{average}_recall": float(r),
                        f"{average}_f1": float(f1)})
    metrics["confusion_matrix"] = confusion_matrix(truth, predicted, labels=list(range(40))).tolist()
    args.output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.predictions, args.output / args.predictions.name)
    shutil.copy2(args.history, args.output / args.history.name)
    with (args.output / "history_clean.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(history)
    with (args.output / "predictions.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["test_index", "true_label", "predicted_label"])
        writer.writerows((i, int(t), int(p)) for i, (t, p) in enumerate(zip(truth, predicted)))
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([r["epoch"] for r in history], [r["train_loss"] for r in history], color="tab:blue")
    ax.set(title="ORL CNN Training Convergence", xlabel="Training epoch", ylabel="Cross-entropy training loss")
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(args.output / "training_convergence.png", dpi=200)
    plt.close(fig)
    print({key: value for key, value in metrics.items() if key != "confusion_matrix"})


if __name__ == "__main__":
    main()
