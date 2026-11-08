
"""CNN model for CIFAR-10 image classification."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from local_code.base_class.method import method

class CIFAR_CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

class Method_CNN_CIFAR(method):
    """CIFAR convolution blocks, classifier, and mini-batch training."""

    def __init__(self, mName="CNN CIFAR",
                 mDescription="Colored object classifier"):
        super().__init__(mName, mDescription)
        self.data = None
        self.history = []

    def run(self, epochs=10, learning_rate=0.001, device="cpu"):

        # Get training and testing data
        X_train = self.data["train"]["X"]
        y_train = self.data["train"]["y"]

        X_test = self.data["test"]["X"]
        y_test = self.data["test"]["y"]

        # Convert to PyTorch tensors
        X_train = torch.as_tensor(
            np.asarray(X_train), dtype=torch.float32
        )
        X_test = torch.as_tensor(
            np.asarray(X_test), dtype=torch.float32
        )

        # Convert NHWC images to NCHW if necessary
        if X_train.ndim == 4 and X_train.shape[-1] == 3:
            X_train = X_train.permute(0, 3, 1, 2)

        if X_test.ndim == 4 and X_test.shape[-1] == 3:
            X_test = X_test.permute(0, 3, 1, 2)

        # Normalize only if pixels are still in 0-255 range
        if X_train.max() > 1:
            X_train = X_train / 255.0

        if X_test.max() > 1:
            X_test = X_test / 255.0

        # Convert labels
        y_train = torch.as_tensor(
            np.asarray(y_train), dtype=torch.long
        ).view(-1)

        y_test = torch.as_tensor(
            np.asarray(y_test), dtype=torch.long
        ).view(-1)

        # Create DataLoaders
        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=64,
            shuffle=True
        )

        test_loader = DataLoader(
            TensorDataset(X_test, y_test),
            batch_size=64,
            shuffle=False
        )

        # Initialize model
        self.model = CIFAR_CNN().to(device)
        model = self.model

        criterion = nn.CrossEntropyLoss()
        #optimizer SGD
        '''optimizer = torch.optim.SGD(
            model.parameters(),
            lr=0.01,
            momentum=0.9,
            weight_decay=1e-4
        )'''
        #optimizer Adagrad
        '''optimizer = torch.optim.Adagrad(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4
        )'''

        #optimzer Adam-accuracy around 82%
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate,
            weight_decay = 1e-4
        )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,
            gamma=0.5
        )
        self.history = []

        # Training
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            correct = 0

            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(images)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(labels)
                correct += (
                        outputs.argmax(dim=1) == labels
                ).sum().item()

            average_loss = total_loss / len(X_train)
            accuracy = correct / len(X_train)

            self.history.append({
                "epoch": epoch + 1,
                "train_loss": average_loss,
                "train_accuracy": accuracy
            })
            scheduler.step()
            print(
                f"Epoch {epoch + 1}/{epochs}, "
                f"Loss: {average_loss:.4f}, "
                f"Accuracy: {accuracy:.4f}"
            )

        # Testing
        model.eval()
        all_predictions = []

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)

                outputs = model(images)
                predictions = outputs.argmax(dim=1)

                all_predictions.append(
                    predictions.cpu()
                )

        predictions = torch.cat(all_predictions)

        return {
            "pred_y": predictions.numpy(),
            "true_y": y_test.numpy()
        }

    def load_checkpoint(self, path):
        """Restore a previously saved CIFAR CNN checkpoint."""
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=True
        )

        if not hasattr(self, "model"):
            self.model = CIFAR_CNN()

        self.model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True
        )

        return checkpoint
