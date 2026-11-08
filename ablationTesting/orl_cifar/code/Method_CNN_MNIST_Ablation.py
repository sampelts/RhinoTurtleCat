'''
CNN for the MNIST hand-written digit dataset: 28x28 grayscale images, 10 classes.

Uses the shared, configurable Method_CNN (Sai) so MNIST can run the same
configuration experiments (step 3-5) as ORL and CIFAR. The baseline keeps the
architecture of the original MNIST model (Andrew): two conv blocks with 16 and
32 filters, a 128-unit dense layer, Adam lr 1e-3, batch size 64, 10 epochs.
Differences from the original MNIST run: inputs are standardized by
Dataset_Loader, and the shared classifier applies dropout 0.5.

Named *_Ablation so it does not collide with Method_CNN_MNIST.py on main.
'''

from local_code.stage_3_code.Method_CNN import Method_CNN


class Method_CNN_MNIST_Ablation(Method_CNN):
    input_shape = (1, 28, 28)
    num_classes = 10

    config = {
        **Method_CNN.config,
        'conv_channels': (16, 32),
        'hidden_dim': 128,
        'batch_size': 64,
        'max_epoch': 10,
    }
