'''
Concrete IO class for the CIFAR-10 colored object dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.dataset import dataset
import numpy as np
import pickle


class Dataset_Loader_CIFAR(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def to_arrays(self, instances):
        X = np.stack([np.asarray(instance['image'], dtype=np.float32) for instance in instances]) / 255.0
        y = np.array([instance['label'] for instance in instances], dtype=np.int64)
        # (N, H, W, C) -> (N, C, H, W), the layout nn.Conv2d expects
        return np.ascontiguousarray(X.transpose(0, 3, 1, 2)), y

    def load(self):
        print('loading data...')
        with open(self.dataset_source_folder_path + self.dataset_source_file_name, 'rb') as f:
            data = pickle.load(f)
        X_train, y_train = self.to_arrays(data['train'])
        X_test, y_test = self.to_arrays(data['test'])

        # standardize each color channel with the training set statistics
        mean = X_train.mean(axis=(0, 2, 3), keepdims=True)
        std = X_train.std(axis=(0, 2, 3), keepdims=True) + 1e-8
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

        print('train:', X_train.shape, 'test:', X_test.shape)
        # Setting_Train_Test_PreSplit expects a (train, test) pair
        return {'X': X_train, 'y': y_train}, {'X': X_test, 'y': y_test}
