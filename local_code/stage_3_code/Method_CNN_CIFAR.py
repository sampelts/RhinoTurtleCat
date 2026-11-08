'''
CNN for the CIFAR-10 colored object dataset: 32x32 RGB images, 10 classes
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.stage_3_code.Method_CNN import Method_CNN


class Method_CNN_CIFAR(Method_CNN):
    input_shape = (3, 32, 32)
    num_classes = 10

    config = {
        **Method_CNN.config,
        'conv_channels': (32, 64, 128),
        'hidden_dim': 256,
        'batch_size': 128,
        'max_epoch': 20,
    }
