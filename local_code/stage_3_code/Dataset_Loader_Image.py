'''
Concrete IO class for a specific dataset
'''
from torch.utils.data import TensorDataset

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.dataset import dataset
import numpy as np
import torch
import pickle
import os

class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    channel_index = None
    label_offset = 0

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load_images(self, samples):
        images = np.stack([np.asarray(s["image"]) for s in samples])
        if images.ndim == 3:        #(N,H,W) -- v> (N,1,H,W)
            images = images[:,None,:,:]
        elif images.ndim == 4:      #(N,H,W,C) --> (N,C,H,W) - pycharm preferred format
            images = images.transpose(0, 3, 1, 2)
        else:
            raise ValueError("Unsupported image dimensions")
        if self.channel_index is not None:
            if not 0 <= self.channel_index <= images.shape[1]:
                raise ValueError("Channel index out of range")
            images = images[:,self.channel_index:self.channel_index+1]
        images = images.astype(np.float32) / 255.0
        return torch.from_numpy(np.ascontiguousarray(images))

    def load_labels(self, samples):
        labels = torch.tensor([s['label'] for s in samples], dtype=torch.long) - self.label_offset
        if labels.min() < 0:
            raise ValueError("Label index out of range")
        return labels

    def load(self):
        print('loading data...')
        path = os.path.join(self.dataset_source_folder_path, self.dataset_source_file_name)
        with open(path, 'rb') as f:
            data = pickle.load(f)
        train_x = self.load_images(data['train'])
        test_x = self.load_images(data['test'])
        train_y = self.load_labels(data['train'])
        test_y = self.load_labels(data['test'])

        mean = train_x.mean(dim=(0,2,3), keepdim=True)
        std = train_x.std(dim=(0,2,3), keepdim=True)
        train_x = (train_x - mean) / std
        test_x = (test_x - mean) / std

        return TensorDataset(train_x, train_y), TensorDataset(test_x, test_y)