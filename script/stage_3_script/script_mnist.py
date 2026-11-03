"""Train on all supplied MNIST training images, then evaluate the test set once."""

# Parse command-line options such as epoch count and output-folder name.
import argparse
# Write the per-epoch history table as CSV.
import csv
# Save run configuration and final metrics in JSON format.
import json
# Configure a writable Matplotlib cache location.
import os
# Build robust paths relative to this repository.
from pathlib import Path
# Seed Python's built-in random generator for reproducibility.
import random
# Validate the requested output-folder name.
import re
# Make repository modules importable when this file is run directly.
import sys
# Measure the total training-and-evaluation runtime.
import time

# Identify the repository root from script/stage_3_script/script_mnist.py.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Put that root first on Python's module search path.
sys.path.insert(0, str(PROJECT_ROOT))
# Keep Matplotlib cache files inside the repository instead of a system folder.
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))

# Import Matplotlib only after its cache directory has been set.
import matplotlib
# Use a noninteractive graphics backend so the script works without a display.
matplotlib.use("Agg")
# Create and save the learning-curve figures.
import matplotlib.pyplot as plt
# Handle arrays, finite-number validation, and random seeding.
import numpy as np
# Train and evaluate the CNN with PyTorch.
import torch
# Compute weighted precision, recall, F1, and the confusion matrix.
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
# Combine arrays into PyTorch datasets and mini-batch data loaders.
from torch.utils.data import DataLoader, TensorDataset

# Import the Stage 3 MNIST-specific dataset loader.
from local_code.stage_3_code.Dataset_Loader_MNIST import Dataset_Loader_MNIST
# Import the Stage 3 MNIST CNN implementation.
from local_code.stage_3_code.Method_CNN_MNIST import Method_CNN_MNIST


# Evaluate a trained model once without updating any of its weights.
def evaluate(model, loader):
    """Compute test metrics without changing model weights."""
    # Switch PyTorch layers to evaluation behavior.
    model.eval()
    # Collect ground-truth labels and predicted labels for all test images.
    truth, predictions = [], []
    # Accumulate summed cross-entropy loss across the full test set.
    loss = 0.0
    # Disable gradients because test evaluation does not need backpropagation.
    with torch.inference_mode():
        # Evaluate every test mini-batch.
        for images, labels in loader:
            # Produce ten digit scores per test image.
            logits = model(images)
            # Add the batch's total cross-entropy loss to the running sum.
            loss += torch.nn.functional.cross_entropy(logits, labels, reduction="sum").item()
            # Save actual digit labels as ordinary Python integers.
            truth.extend(labels.tolist())
            # Save the highest-scoring predicted digit for every image.
            predictions.extend(logits.argmax(1).tolist())
    # Calculate support-weighted metrics across all ten digit classes.
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, predictions, labels=list(range(10)), average="weighted", zero_division=0)
    # Count each test image where prediction equals the true label.
    correct = sum(actual == predicted for actual, predicted in zip(truth, predictions))
    # Return every report-relevant metric in a JSON-serializable dictionary.
    return {"samples": len(truth), "correct": correct, "loss": loss / len(truth),
            "accuracy": correct / len(truth), "weighted_precision": float(precision),
            "weighted_recall": float(recall), "weighted_f1": float(f1),
            "confusion_matrix": confusion_matrix(truth, predictions, labels=list(range(10))).tolist()}


# Configure, train, evaluate, and save one reproducible MNIST experiment.
def main(argv=None):
    # Create the command-line parser using this module's description text.
    parser = argparse.ArgumentParser(description=__doc__)
    # Let the user name the result folder for this run.
    parser.add_argument("--run-name", default="mnist_simple_1")
    # Let the user select how many full training passes to make.
    parser.add_argument("--epochs", type=int, default=10)
    # Let the user select mini-batch size.
    parser.add_argument("--batch-size", type=int, default=64)
    # Let the user select Adam's learning rate.
    parser.add_argument("--learning-rate", type=float, default=0.001)
    # Allow the two kernel sizes used for simple architecture comparisons.
    parser.add_argument("--kernel-size", type=int, choices=(3, 5), default=3)
    # Allow either max or average pooling.
    parser.add_argument("--pooling", choices=("max", "avg"), default="max")
    # Keep the documented reproducible run on CPU.
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    # Read the user's command-line values, or supplied test values in argv.
    args = parser.parse_args(argv)
    # Reject nonpositive or nonfinite optimization settings.
    if args.epochs < 1 or args.batch_size < 1 or not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("epochs, batch size, and learning rate must be positive")
    # Reject output-folder names that could escape the intended results directory.
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.run_name):
        parser.error("Use letters, digits, underscores, or hyphens for the run name")
    # Choose a dedicated result folder beneath the Stage 3 MNIST results directory.
    output = PROJECT_ROOT / "result" / "stage_3_result" / "mnist" / args.run_name
    # Protect an earlier result set from being overwritten accidentally.
    if output.exists():
        parser.error("This run already exists; choose a new --run-name")
    # Create the nested result folder for this new run.
    output.mkdir(parents=True)

    # Seed standard Python randomness.
    random.seed(42)
    # Seed NumPy randomness.
    np.random.seed(42)
    # Seed PyTorch parameter initialization and random operations.
    torch.manual_seed(42)
    # Request deterministic PyTorch operations for reproducible results.
    torch.use_deterministic_algorithms(True)
    # Use one CPU thread to reduce platform-dependent ordering variation.
    torch.set_num_threads(1)
    # Start the wall-clock timer before loading and training data.
    started = time.perf_counter()
    # Create the MNIST loader with its default repository data location.
    data = Dataset_Loader_MNIST()
    # Load all 60,000 supplied training examples.
    training = data.load("train")
    # Wrap NumPy arrays in a shuffled, reproducible PyTorch mini-batch loader.
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(training["X"]), torch.from_numpy(training["y"])),
        batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(42))
    # Record settings needed to reproduce and describe this exact run.
    config = {**vars(args), "seed": 42, "training_samples": len(training["y"]),
              "input_shape": [1, 28, 28], "conv_channels": [16, 32], "hidden_sizes": [128],
              "pixel_range": "0..255 (no scaling)", "optimizer": "Adam",
              "selection_rule": "final epoch; fixed before test evaluation", "validation_samples": 0}
    # Write the configuration before training begins.
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    # Build the CNN, matching padding to the selected odd kernel size.
    model = Method_CNN_MNIST(kernel_size=args.kernel_size, padding=args.kernel_size // 2, pooling=args.pooling)
    # Print the exact number of training images for a clear console record.
    print(f"Training on all {len(training['y']):,} MNIST training images", flush=True)
    # Train the CNN and collect loss and accuracy for every epoch.
    history = model.run(train_loader, args.epochs, args.learning_rate)
    # Save model weights locally so this exact trained model can be restored.
    torch.save({"model_state_dict": model.state_dict(), "config": config}, output / "final.pt")
    # Open the history CSV for writing one record per epoch.
    with (output / "history.csv").open("w", newline="") as stream:
        # Make CSV columns from the keys returned by the first epoch.
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        # Write the CSV header row.
        writer.writeheader()
        # Write every epoch's loss and accuracy record.
        writer.writerows(history)
    # Create a side-by-side loss and accuracy learning-curve figure.
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    # Draw both tracked metrics using the same epoch values on the x-axis.
    for axis, metric in zip(axes, ("train_loss", "train_accuracy")):
        # Plot the selected metric as a line with visible epoch markers.
        axis.plot([row["epoch"] for row in history], [row[metric] for row in history], "o-")
        # Label the axis and create a readable plot title.
        axis.set(xlabel="Epoch", ylabel=metric, title=metric.replace("_", " "))
        # Add a light grid to help read values at each epoch.
        axis.grid(alpha=0.3)
    # Save the combined learning-curve image for the report or repository.
    figure.savefig(output / "learning_curve.png", dpi=150)
    # Release the figure memory before making the next image.
    plt.close(figure)

    # Create the report-required epoch-versus-loss convergence plot.
    figure, axis = plt.subplots(figsize=(7, 4))
    # Plot training loss as a labeled blue line with circular markers.
    axis.plot([row["epoch"] for row in history], [row["train_loss"] for row in history],
              "o-", color="tab:blue", label="Training loss")
    # State exactly what each visual element represents.
    axis.set(title="MNIST CNN Training Convergence", xlabel="Training epoch",
             ylabel="Cross-entropy training loss")
    # Label every integer epoch on the x-axis.
    axis.set_xticks([row["epoch"] for row in history])
    # Add a light value-reading grid.
    axis.grid(alpha=0.3)
    # Show the line's label in a legend.
    axis.legend()
    # Prevent labels from overlapping or being cut off in the saved image.
    figure.tight_layout()
    # Save the report-ready convergence image at a higher resolution.
    figure.savefig(output / "training_convergence.png", dpi=200)
    # Release the second figure's memory.
    plt.close(figure)

    # Load the separate, untouched supplied MNIST test split after training.
    test = data.load("test")
    # Wrap test arrays in ordered mini-batches without shuffling.
    test_loader = DataLoader(TensorDataset(torch.from_numpy(test["X"]), torch.from_numpy(test["y"])),
                             batch_size=args.batch_size, shuffle=False)
    # Assemble model size, runtime, and final untouched-test results.
    metrics = {"training_samples": len(training["y"]), "epochs": args.epochs,
               "parameter_count": sum(p.numel() for p in model.parameters()),
               "test": evaluate(model, test_loader), "elapsed_seconds": time.perf_counter() - started}
    # Write all metrics to JSON for GitHub and report use.
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    # Display the same metrics in the PyCharm or terminal console.
    print(json.dumps(metrics, indent=2))
    # Tell the user where all files from this run were saved.
    print(f"Results saved to {output}")


# Run main only when this file is executed directly, not merely imported.
if __name__ == "__main__":
    # Start the full training-and-evaluation workflow.
    main()
