'''
CNN for the ORL face dataset: 112x92 grayscale images, 40 people
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.stage_3_code.Method_CNN import Method_CNN


class Method_CNN_ORL(Method_CNN):
    # one channel is enough since R=G=B for the face images
    input_shape = (1, 112, 92)
    num_classes = 40

    # only 360 training images (9 per person), so a small network,
    # small batches and more epochs
    config = {
        **Method_CNN.config,
        'conv_channels': (16, 32, 64),
        'hidden_dim': 256,
        'batch_size': 32,
        'max_epoch': 50,
    }
