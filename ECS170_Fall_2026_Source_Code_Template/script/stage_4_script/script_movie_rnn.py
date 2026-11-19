
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Set project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / 'data' / 'stage_4_data' / 'text_classification'

from local_code.stage_4_code.Dataset_Loader_Movie import (
    Dataset_Loader,
    MoviesDataset
)
from local_code.stage_4_code.Method_RNN_Movie import Movie_RNN
from local_code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy


def evaluate_model(model, loader, device):
    model.eval()

    true_y = []
    pred_y = []

    with torch.no_grad():
        for x, lengths, y in loader:
            x = x.to(device)

            outputs = model(x, lengths)

            predictions = (outputs >= 0).long()

            true_y.extend(y.tolist())
            pred_y.extend(predictions.cpu().tolist())

    evaluator = Evaluate_Accuracy(
        num_classes=2,
        label_start=0
    )

    evaluator.data = {
        'true_y': true_y,
        'pred_y': pred_y
    }

    return evaluator.evaluate()


def main():

    # Set random seeds
    torch.manual_seed(2)

    # Select device
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    print("Using device:", device)

    # Load dataset
    dataset = Dataset_Loader('IMDb', 'Movie Reviews')

    dataset.dataset_source_folder_path = str(DATA_DIR)
    dataset.dataset_source_file_name = 'movie_encoded.pt'

    datasets = dataset.load()

    print("Vocabulary size:", dataset.vocab_size)
    print("Training samples:", len(datasets['train']))
    print("Validation samples:", len(datasets['val']))
    print("Testing samples:", len(datasets['test']))

    # Create DataLoaders
    train_loader = DataLoader(
        datasets['train'],
        batch_size=64,
        shuffle=True,
        collate_fn=MoviesDataset.collate_fn
    )

    val_loader = DataLoader(
        datasets['val'],
        batch_size=64,
        shuffle=False,
        collate_fn=MoviesDataset.collate_fn
    )

    test_loader = DataLoader(
        datasets['test'],
        batch_size=64,
        shuffle=False,
        collate_fn=MoviesDataset.collate_fn
    )

    # Initialize RNN
    model = Movie_RNN(
        vocab_size=dataset.vocab_size
    ).to(device)

    # Loss function
    criterion = nn.BCEWithLogitsLoss()

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    # Train model
    epochs = 10
    best_val_accuracy = -1
    best_model_state = None

    for epoch in range(epochs):

        model.train()
        total_loss = 0
        total_samples = 0

        for x, lengths, y in train_loader:

            x = x.to(device)
            y = y.float().to(device)

            # Reset gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(x, lengths)

            # Calculate loss
            loss = criterion(outputs, y)

            # Backpropagation
            loss.backward()

            # Prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )

            # Update weights
            optimizer.step()

            total_loss += loss.item() * y.size(0)
            total_samples += y.size(0)

        avg_loss = total_loss / total_samples

        # Evaluate validation accuracy
        val_results = evaluate_model(
            model, val_loader, device
        )

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"Val Accuracy: {val_results['accuracy']:.4f} | "
            f"Val F1: {val_results['f1_macro']:.4f}"
        )

        # Save best model weights in memory
        if val_results['accuracy'] > best_val_accuracy:
            best_val_accuracy = val_results['accuracy']
            best_model_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    # Restore best validation model
    model.load_state_dict(best_model_state)

    # Final evaluation on test data
    test_results = evaluate_model(
        model, test_loader, device
    )

    print("\nFinal Test Results")
    print("------------------")

    print("Accuracy:", test_results['accuracy'])
    print("Precision:", test_results['precision_macro'])
    print("Recall:", test_results['recall_macro'])
    print("F1 Score:", test_results['f1_macro'])


if __name__ == '__main__':
    main()
