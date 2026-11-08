'''
CNN for the CIFAR-10 colored object dataset: 32x32 RGB images, 10 classes.
The architecture is configurable (depth, kernel, padding, stride, pooling, ...)
so script_cnn_cifar.py can run the configuration experiments.
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
from local_code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np


class Method_CNN_CIFAR(method, nn.Module):
    data = None
    history = None
    # input shape (channels, height, width) and number of classes
    input_shape = (3, 32, 32)
    num_classes = 10

    # default configuration; any key can be overridden through the constructor
    config = {
        'conv_channels': (32, 64, 128),  # one conv block per entry -> controls model depth
        'kernel_size': 3,
        'padding': 1,
        'stride': 1,
        'pool': 'max',              # 'max', 'avg' or None
        'batch_norm': False,
        'hidden_dim': 256,          # size of the fully connected hidden layer
        'dropout': 0.5,
        'loss': 'cross_entropy',    # 'cross_entropy' or 'multi_margin'
        'optimizer': 'adam',        # 'adam' or 'sgd'
        'learning_rate': 1e-3,
        'weight_decay': 0.0,
        'batch_size': 128,
        'max_epoch': 20,
    }

    def __init__(self, mName, mDescription, **overrides):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        unknown = set(overrides) - set(self.config)
        if unknown:
            raise ValueError('unknown config keys: ' + str(sorted(unknown)))
        self.config = {**self.config, **overrides}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        c = self.config

        # convolution blocks: conv -> (batch norm) -> ReLU -> (pooling)
        layers = []
        in_channels = self.input_shape[0]
        for out_channels in c['conv_channels']:
            layers.append(nn.Conv2d(in_channels, out_channels, c['kernel_size'], stride=c['stride'], padding=c['padding']))
            if c['batch_norm']:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
            if c['pool'] == 'max':
                layers.append(nn.MaxPool2d(2))
            elif c['pool'] == 'avg':
                layers.append(nn.AvgPool2d(2))
            in_channels = out_channels
        self.features = nn.Sequential(*layers)

        # pass a dummy image through the conv stack to find the flattened size
        with torch.no_grad():
            flat_dim = self.features(torch.zeros(1, *self.input_shape)).numel()
        print('conv output size:', flat_dim)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, c['hidden_dim']),
            nn.ReLU(),
            nn.Dropout(c['dropout']),
            nn.Linear(c['hidden_dim'], self.num_classes),
        )
        self.to(self.device)

    def forward(self, x):
        '''Forward propagation'''
        # we use CrossEntropyLoss / MultiMarginLoss, so no softmax here
        return self.classifier(self.features(x))

    # note: the template's train(X, y) shadows nn.Module.train(mode), so self.eval() would break;
    # we switch dropout / batch norm modes with nn.Module.train(self, mode) directly instead

    def predict(self, X, y=None, loss_function=None):
        '''Batched inference; returns predictions and (if y is given) the mean loss'''
        nn.Module.train(self, False)
        preds, total_loss = [], 0.0
        with torch.no_grad():
            for start in range(0, len(X), 512):
                x_batch = torch.from_numpy(X[start:start + 512]).to(self.device)
                y_pred = self.forward(x_batch)
                preds.append(y_pred.argmax(1).cpu())
                if y is not None:
                    y_batch = torch.from_numpy(y[start:start + 512]).long().to(self.device)
                    total_loss += loss_function(y_pred, y_batch).item() * len(x_batch)
        preds = torch.cat(preds).numpy()
        return preds, (total_loss / len(X) if y is not None else None)

    def train(self, X, y, X_test=None, y_test=None):
        c = self.config
        if c['optimizer'] == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=c['learning_rate'], momentum=0.9, weight_decay=c['weight_decay'])
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=c['learning_rate'], weight_decay=c['weight_decay'])
        if c['loss'] == 'multi_margin':
            loss_function = nn.MultiMarginLoss()
        else:
            loss_function = nn.CrossEntropyLoss()
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        for epoch in range(c['max_epoch']):
            nn.Module.train(self, True)
            # shuffle and split the training set into mini-batches
            permutation = np.random.permutation(len(X))
            total_loss, train_preds = 0.0, np.empty(len(X), dtype=np.int64)
            for start in range(0, len(X), c['batch_size']):
                idx = permutation[start:start + c['batch_size']]
                x_batch = torch.from_numpy(X[idx]).to(self.device)
                y_batch = torch.from_numpy(y[idx]).long().to(self.device)

                y_pred = self.forward(x_batch)
                train_loss = loss_function(y_pred, y_batch)
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                total_loss += train_loss.item() * len(idx)
                train_preds[idx] = y_pred.argmax(1).cpu().numpy()

            accuracy_evaluator.data = {'true_y': y, 'pred_y': train_preds}
            train_metrics = accuracy_evaluator.evaluate()
            record = {
                'Epoch': epoch,
                'Train_Loss': total_loss / len(X),
                'Train_Accuracy': train_metrics['accuracy'],
                'Train_F1_Macro': train_metrics['f1_macro'],
            }
            # the test set is only evaluated to draw the learning curves, never to pick a model
            if X_test is not None:
                test_preds, test_loss = self.predict(X_test, y_test, loss_function)
                accuracy_evaluator.data = {'true_y': y_test, 'pred_y': test_preds}
                record['Test_Loss'] = test_loss
                record['Test_Accuracy'] = accuracy_evaluator.evaluate()['accuracy']
            if self.history:
                self.history.record(**record)
            print(', '.join(k + ': ' + (f'{v:.4f}' if isinstance(v, float) else str(v)) for k, v in record.items()))
        if self.history:
            self.history.save()

    def test(self, X):
        # do the testing, and return the result
        return self.predict(X)[0]

    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'],
                   self.data['test']['X'], self.data['test']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}
