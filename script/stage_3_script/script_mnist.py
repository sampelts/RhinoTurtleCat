"""Train on all supplied MNIST training images, then evaluate the test set once."""

import argparse
import csv
import json
import os
from pathlib import Path
import random
import re
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset

from local_code.stage_3_code.Dataset_Loader_MNIST import Dataset_Loader_MNIST
from local_code.stage_3_code.Method_CNN_MNIST import Method_CNN_MNIST


def evaluate(model, loader):
    """Compute test metrics without changing model weights."""
    model.eval()
    truth, predictions = [], []
    loss = 0.0
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images)
            loss += torch.nn.functional.cross_entropy(logits, labels, reduction="sum").item()
            truth.extend(labels.tolist())
            predictions.extend(logits.argmax(1).tolist())
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, predictions, labels=list(range(10)), average="weighted", zero_division=0)
    correct = sum(actual == predicted for actual, predicted in zip(truth, predictions))
    return {"samples": len(truth), "correct": correct, "loss": loss / len(truth),
            "accuracy": correct / len(truth), "weighted_precision": float(precision),
            "weighted_recall": float(recall), "weighted_f1": float(f1),
            "confusion_matrix": confusion_matrix(truth, predictions, labels=list(range(10))).tolist()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="mnist_simple_1")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--kernel-size", type=int, choices=(3, 5), default=3)
    parser.add_argument("--pooling", choices=("max", "avg"), default="max")
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("epochs, batch size, and learning rate must be positive")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.run_name):
        parser.error("Use letters, digits, underscores, or hyphens for the run name")
    output = PROJECT_ROOT / "result" / "stage_3_result" / "mnist" / args.run_name
    if output.exists():
        parser.error("This run already exists; choose a new --run-name")
    output.mkdir(parents=True)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    started = time.perf_counter()
    data = Dataset_Loader_MNIST()
    training = data.load("train")
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(training["X"]), torch.from_numpy(training["y"])),
        batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(42))
    config = {**vars(args), "seed": 42, "training_samples": len(training["y"]),
              "input_shape": [1, 28, 28], "conv_channels": [16, 32], "hidden_sizes": [128],
              "pixel_range": "0..255 (no scaling)", "optimizer": "Adam",
              "selection_rule": "final epoch; fixed before test evaluation", "validation_samples": 0}
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    model = Method_CNN_MNIST(kernel_size=args.kernel_size, padding=args.kernel_size // 2, pooling=args.pooling)
    print(f"Training on all {len(training['y']):,} MNIST training images", flush=True)
    history = model.run(train_loader, args.epochs, args.learning_rate)
    torch.save({"model_state_dict": model.state_dict(), "config": config}, output / "final.pt")
    with (output / "history.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for axis, metric in zip(axes, ("train_loss", "train_accuracy")):
        axis.plot([row["epoch"] for row in history], [row[metric] for row in history], "o-")
        axis.set(xlabel="Epoch", ylabel=metric, title=metric.replace("_", " "))
        axis.grid(alpha=0.3)
    figure.savefig(output / "learning_curve.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot([row["epoch"] for row in history], [row["train_loss"] for row in history],
              "o-", color="tab:blue", label="Training loss")
    axis.set(title="MNIST CNN Training Convergence", xlabel="Training epoch",
             ylabel="Cross-entropy training loss")
    axis.set_xticks([row["epoch"] for row in history])
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "training_convergence.png", dpi=200)
    plt.close(figure)

    test = data.load("test")
    test_loader = DataLoader(TensorDataset(torch.from_numpy(test["X"]), torch.from_numpy(test["y"])),
                             batch_size=args.batch_size, shuffle=False)
    metrics = {"training_samples": len(training["y"]), "epochs": args.epochs,
               "parameter_count": sum(p.numel() for p in model.parameters()),
               "test": evaluate(model, test_loader), "elapsed_seconds": time.perf_counter() - started}
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    print(f"Results saved to {output}")


if __name__ == "__main__":
    main()
