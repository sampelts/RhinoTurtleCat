'''
Concrete IO class for the stage 3 image datasets (ORL, CIFAR, MNIST)
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.dataset import dataset
import numpy as np
import pickle


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    # ORL faces are stored as RGB with R=G=B, so one channel is enough
    grayscale = False

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def to_arrays(self, instances):
        X = np.stack([np.asarray(instance['image'], dtype=np.float32) for instance in instances]) / 255.0
        y = np.array([instance['label'] for instance in instances])
        if X.ndim == 3:
            # (N, H, W) -> (N, 1, H, W)
            X = X[:, None]
        else:
            # (N, H, W, C) -> (N, C, H, W), the layout nn.Conv2d expects
            X = X.transpose(0, 3, 1, 2)
            if self.grayscale:
                X = X[:, :1]
        return np.ascontiguousarray(X), y

    def load(self):
        print('loading data...')
        with open(self.dataset_source_folder_path + self.dataset_source_file_name, 'rb') as f:
            data = pickle.load(f)
        X_train, y_train = self.to_arrays(data['train'])
        X_test, y_test = self.to_arrays(data['test'])

        # standardize each channel with the training set statistics
        mean = X_train.mean(axis=(0, 2, 3), keepdims=True)
        std = X_train.std(axis=(0, 2, 3), keepdims=True) + 1e-8
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

        # map labels to 0..K-1 (ORL labels start at 1)
        classes = np.unique(y_train)
        y_train = np.searchsorted(classes, y_train)
        y_test = np.searchsorted(classes, y_test)

        print('train:', X_train.shape, 'test:', X_test.shape, 'classes:', len(classes))
        return {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}
