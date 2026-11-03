"""A configurable CNN implemented specifically for 1 x 28 x 28 MNIST digits."""

# Use PyTorch for tensor calculations, training, and checkpoint loading.
import torch
# Import PyTorch neural-network layers and containers.
from torch import nn

# Inherit from the instructor's required method base class.
from local_code.base_class.method import method


# Define the MNIST CNN, combining the base method and PyTorch module interfaces.
class Method_CNN_MNIST(method, nn.Module):
    """MNIST convolution blocks, digit classifier, and mini-batch training."""

    # Set up configurable convolutional and dense layers for MNIST images.
    def __init__(self, mName="CNN MNIST", mDescription="Grayscale digit classifier", *,
                 conv_channels=(16, 32), hidden_sizes=(128,), kernel_size=3,
                 padding=1, stride=1, pooling="max"):
        # Initialize the instructor-provided method metadata.
        method.__init__(self, mName, mDescription)
        # Initialize PyTorch's parameter-registration machinery.
        nn.Module.__init__(self)
        # Ensure at least one positive convolution-filter count was supplied.
        if not conv_channels or any(type(n) is not int or n <= 0 for n in conv_channels):
            raise ValueError("conv_channels must contain positive integers")
        # Ensure all requested dense-layer widths are positive integers.
        if any(type(n) is not int or n <= 0 for n in hidden_sizes):
            raise ValueError("hidden_sizes must contain positive integers")
        # Validate kernel, padding, and stride settings before building layers.
        for name, value, minimum in (("kernel_size", kernel_size, 1), ("padding", padding, 0), ("stride", stride, 1)):
            # Reject a setting that would be invalid for PyTorch convolution.
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        # Limit pooling to the supported choices.
        if pooling not in {"max", "avg", "none"}:
            raise ValueError("pooling must be max, avg, or none")

        # MNIST images have one channel and spatial dimensions 28 by 28.
        self.input_shape = (1, 28, 28)
        # Track feature-map dimensions as convolution blocks are built.
        channels, height, width = self.input_shape
        # Accumulate the convolution, activation, and pooling layers here.
        layers = []
        # Create one convolutional block for every requested output-channel count.
        for output_channels in conv_channels:
            # Calculate height after the upcoming convolution.
            height = (height + 2 * padding - kernel_size) // stride + 1
            # Calculate width after the upcoming convolution.
            width = (width + 2 * padding - kernel_size) // stride + 1
            # Halve each spatial dimension when 2-by-2 pooling is used.
            if pooling != "none":
                height, width = height // 2, width // 2
            # Stop if the requested design would shrink a feature map to zero.
            if min(height, width) < 1:
                raise ValueError("CNN configuration collapses the MNIST spatial dimensions")
            # Add convolution followed by the ReLU nonlinearity.
            layers.extend([nn.Conv2d(channels, output_channels, kernel_size, stride=stride, padding=padding), nn.ReLU()])
            # Add max or average pooling unless pooling was disabled.
            if pooling != "none":
                layers.append(nn.MaxPool2d(2) if pooling == "max" else nn.AvgPool2d(2))
            # Use this block's output-channel count as the next block's input count.
            channels = output_channels
        # Register the completed convolutional feature extractor.
        self.features = nn.Sequential(*layers)

        # Start the classifier by flattening each final feature map to one vector.
        dense = [nn.Flatten()]
        # Calculate the number of flattened features after all convolution blocks.
        previous = channels * height * width
        # Add every requested dense layer and its ReLU activation.
        for size in hidden_sizes:
            dense.extend([nn.Linear(previous, size), nn.ReLU()])
            # Feed the next dense layer from this layer's output width.
            previous = size
        # Add ten unnormalized output scores, one for each MNIST digit.
        dense.append(nn.Linear(previous, 10))
        # Register the completed classification portion of the CNN.
        self.classifier = nn.Sequential(*dense)

    # Define the forward calculation used for training and prediction.
    def forward(self, images):
        # Validate that the caller supplied a batch of channel-first MNIST images.
        if images.ndim != 4 or tuple(images.shape[1:]) != self.input_shape:
            raise ValueError(f"Expected MNIST images shaped N x {self.input_shape}, got {tuple(images.shape)}")
        # Extract convolutional features and classify them into ten digit logits.
        return self.classifier(self.features(images))

    # Train the model for one full pass through the provided mini-batch loader.
    def train_epoch(self, loader, optimizer, device):
        """Train this MNIST model for one full pass through shuffled mini-batches."""
        # Put PyTorch layers such as dropout or batch norm into training mode.
        self.train()
        # Use multiclass cross-entropy between digit logits and integer labels.
        loss_function = nn.CrossEntropyLoss()
        # Initialize aggregates for epoch-average loss and accuracy.
        total_loss, correct, count = 0.0, 0, 0
        # Process every shuffled mini-batch exactly once.
        for images, labels in loader:
            # Move the batch to the selected computation device.
            images, labels = images.to(device), labels.to(device)
            # Clear gradients from the previous mini-batch.
            optimizer.zero_grad(set_to_none=True)
            # Compute ten class scores for every image in the mini-batch.
            logits = self(images)
            # Calculate the batch's cross-entropy loss.
            loss = loss_function(logits, labels)
            # Halt with a clear error if a bad configuration produces NaN or infinity.
            if not torch.isfinite(loss).item():
                raise RuntimeError("MNIST training loss became non-finite; inspect the configuration")
            # Backpropagate the loss through all model parameters.
            loss.backward()
            # Apply Adam's parameter update.
            optimizer.step()
            # Add this batch's summed loss to the epoch total.
            total_loss += loss.item() * len(labels)
            # Count predictions whose highest-scoring class is the true digit.
            correct += (logits.argmax(dim=1) == labels).sum().item()
            # Count the examples seen so far this epoch.
            count += len(labels)
        # Return loss and accuracy averaged over all training images.
        return {"loss": total_loss / count, "accuracy": correct / count}

    # Fit all supplied training images for a fixed number of epochs.
    def run(self, train_loader, epochs=10, learning_rate=0.001, device="cpu"):
        """Fit every training image for a fixed number of epochs."""
        # Move registered model parameters to the requested device.
        self.to(device)
        # Create Adam with the report's selected learning rate.
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        # Store one loss-and-accuracy record per epoch for saved learning curves.
        history = []
        # Repeat full training-set passes from epoch 1 through the requested count.
        for epoch in range(1, epochs + 1):
            # Train once through every mini-batch.
            training = self.train_epoch(train_loader, optimizer, device)
            # Save the epoch number and aggregate values for later CSV output.
            history.append({"epoch": epoch, "train_loss": training["loss"],
                            "train_accuracy": training["accuracy"]})
            # Print concise live progress for a terminal or PyCharm console.
            print(
                f"MNIST epoch {epoch:02d}/{epochs}: "
                f"loss={training['loss']:.6f}, accuracy={training['accuracy']:.4%}",
                flush=True,
            )
        # Return all epoch records to the runner script.
        return history

    # Restore a previously saved model checkpoint for later use.
    def load_checkpoint(self, path):
        # Load the checkpoint onto CPU without loading unsafe arbitrary objects.
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        # Copy the saved parameter tensors into this matching model architecture.
        self.load_state_dict(checkpoint["model_state_dict"], strict=True)
        # Return the full checkpoint dictionary, including its saved configuration.
        return checkpoint
