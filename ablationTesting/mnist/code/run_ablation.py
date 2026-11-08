"""Reproduce either selected MNIST ablation or evaluate its saved checkpoint."""

import argparse
import csv
import importlib
import json
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[3]
MNIST = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset

from ablationTesting.mnist.code.Dataset_Loader_MNIST import Dataset_Loader_MNIST


def evaluate(model, loader):
    """Compute test metrics with the same definitions as the original study."""
    model.eval()
    truth, predictions, loss = [], [], 0.0
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images)
            loss += torch.nn.functional.cross_entropy(logits, labels, reduction="sum").item()
            truth.extend(labels.tolist())
            predictions.extend(logits.argmax(1).tolist())
    p, r, f1, _ = precision_recall_fscore_support(
        truth, predictions, labels=list(range(10)), average="weighted", zero_division=0)
    correct = sum(t == p for t, p in zip(truth, predictions))
    return {"samples": len(truth), "correct": correct, "loss": loss / len(truth),
            "accuracy": correct / len(truth), "weighted_precision": float(p),
            "weighted_recall": float(r), "weighted_f1": float(f1),
            "confusion_matrix": confusion_matrix(truth, predictions, labels=list(range(10))).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("best_average_pool", "worst_shallow"), required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "stage_3_data")
    parser.add_argument("--output", type=Path, help="New output folder, required for training")
    parser.add_argument("--evaluate-only", action="store_true", help="Evaluate the included checkpoint without training")
    args = parser.parse_args()
    if not args.evaluate_only and args.output is None:
        parser.error("Training requires --output pointing to a new folder")
    if args.output is not None and args.output.exists():
        parser.error("Output folder exists; choose a new path to preserve saved results")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    module = importlib.import_module(f"ablationTesting.mnist.{args.variant}.Method_CNN_MNIST")
    model = module.Method_CNN_MNIST()
    data = Dataset_Loader_MNIST(data_dir=args.data_dir)
    saved = MNIST / args.variant / "results"
    config = json.loads((saved / "config.json").read_text())
    history = []
    if args.evaluate_only:
        checkpoint = torch.load(saved / "final.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    else:
        train = data.load("train")
        if len(train["y"]) != 60000:
            raise ValueError("Expected the complete 60,000-image MNIST training split")
        loader = DataLoader(TensorDataset(torch.from_numpy(train["X"]), torch.from_numpy(train["y"])),
                            batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(42))
        history = model.run(loader, epochs=10, learning_rate=0.001)
    test = data.load("test")
    if len(test["y"]) != 10000:
        raise ValueError("Expected the complete 10,000-image MNIST test split")
    loader = DataLoader(TensorDataset(torch.from_numpy(test["X"]), torch.from_numpy(test["y"])), batch_size=64)
    metrics = evaluate(model, loader)
    print(json.dumps(metrics, indent=2))
    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        if history:
            (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
            torch.save({"model_state_dict": model.state_dict(), "model_config": config["model"]}, args.output / "final.pt")
            with (args.output / "history.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(history[0]), lineterminator="\n")
                writer.writeheader()
                writer.writerows(history)
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(range(1, 11), [r["train_loss"] for r in history], "o-")
            ax.set(title=f"MNIST: {args.variant}", xlabel="Training epoch", ylabel="Cross-entropy training loss")
            ax.grid(alpha=.25)
            fig.tight_layout()
            fig.savefig(args.output / "training_convergence.png", dpi=200)
            plt.close(fig)


if __name__ == "__main__":
    main()
