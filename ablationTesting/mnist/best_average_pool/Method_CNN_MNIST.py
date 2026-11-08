"""A configurable CNN implemented specifically for 1 x 28 x 28 MNIST digits."""

import torch
from torch import nn

from local_code.base_class.method import method


class Method_CNN_MNIST(method, nn.Module):
    """MNIST convolution blocks, digit classifier, and mini-batch training."""

    def __init__(self, mName="CNN MNIST", mDescription="Grayscale digit classifier", *,
                 conv_channels=(16, 32), hidden_sizes=(128,), kernel_size=3,
                 padding=1, stride=1, pooling="avg"):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        if not conv_channels or any(type(n) is not int or n <= 0 for n in conv_channels):
            raise ValueError("conv_channels must contain positive integers")
        if any(type(n) is not int or n <= 0 for n in hidden_sizes):
            raise ValueError("hidden_sizes must contain positive integers")
        for name, value, minimum in (("kernel_size", kernel_size, 1), ("padding", padding, 0), ("stride", stride, 1)):
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if pooling not in {"max", "avg", "none"}:
            raise ValueError("pooling must be max, avg, or none")

        self.input_shape = (1, 28, 28)
        channels, height, width = self.input_shape
        layers = []
        for output_channels in conv_channels:
            height = (height + 2 * padding - kernel_size) // stride + 1
            width = (width + 2 * padding - kernel_size) // stride + 1
            if pooling != "none":
                height, width = height // 2, width // 2
            if min(height, width) < 1:
                raise ValueError("CNN configuration collapses the MNIST spatial dimensions")
            layers.extend([nn.Conv2d(channels, output_channels, kernel_size, stride=stride, padding=padding), nn.ReLU()])
            if pooling != "none":
                layers.append(nn.MaxPool2d(2) if pooling == "max" else nn.AvgPool2d(2))
            channels = output_channels
        self.features = nn.Sequential(*layers)

        dense = [nn.Flatten()]
        previous = channels * height * width
        for size in hidden_sizes:
            dense.extend([nn.Linear(previous, size), nn.ReLU()])
            previous = size
        dense.append(nn.Linear(previous, 10))
        self.classifier = nn.Sequential(*dense)

    def forward(self, images):
        if images.ndim != 4 or tuple(images.shape[1:]) != self.input_shape:
            raise ValueError(f"Expected MNIST images shaped N x {self.input_shape}, got {tuple(images.shape)}")
        return self.classifier(self.features(images))

    def train_epoch(self, loader, optimizer, device):
        """Train this MNIST model for one full pass through shuffled mini-batches."""
        self.train()
        loss_function = nn.CrossEntropyLoss()
        total_loss, correct, count = 0.0, 0, 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = self(images)
            loss = loss_function(logits, labels)
            if not torch.isfinite(loss).item():
                raise RuntimeError("MNIST training loss became non-finite; inspect the configuration")
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            count += len(labels)
        return {"loss": total_loss / count, "accuracy": correct / count}

    def run(self, train_loader, epochs=10, learning_rate=0.001, device="cpu"):
        """Fit every training image for a fixed number of epochs."""
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        history = []
        for epoch in range(1, epochs + 1):
            training = self.train_epoch(train_loader, optimizer, device)
            history.append({"epoch": epoch, "train_loss": training["loss"],
                            "train_accuracy": training["accuracy"]})
            print(
                f"MNIST epoch {epoch:02d}/{epochs}: "
                f"loss={training['loss']:.6f}, accuracy={training['accuracy']:.4%}",
                flush=True,
            )
        return history

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        self.load_state_dict(checkpoint["model_state_dict"], strict=True)
        return checkpoint
