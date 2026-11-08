from local_code.base_class.method import method
from local_code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
from torch.utils.data import DataLoader

class Method_CNN_Base(method, nn.Module):
    # expected format: {'train': TensorDataset, 'test': TensorDataset}
    # images: (N, C, H, W) float32, labels: (N,) long
    data = None
    history = None
    max_epoch = 20
    batch_size = 128
    learning_rate = 1e-3
    # 'Adam', 'SGD', 'RMSprop'
    optimizer_name = 'Adam'
    # 'cross_entropy', 'label_smoothing', 'multi_margin'
    loss_name = 'cross_entropy'

    def __init__(self, mName, mDescription):
        # both parents are initialised explicitly; nn.Module must be initialised
        # before a subclass assigns any layers
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

    def forward(self, x):
        raise NotImplementedError('Subclasses must implement forward()')

    def _build_optimizer(self):
        opt_name = self.optimizer_name.lower()
        if opt_name == 'adam':
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        elif opt_name == 'sgd':
            return torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        elif opt_name == 'rmsprop':
            return torch.optim.RMSprop(self.parameters(), lr=self.learning_rate)
        raise ValueError('Unknown optimizer')

    def _build_loss(self):
        name = self.loss_name.lower()
        if name == 'cross_entropy':
            return nn.CrossEntropyLoss()
        elif name == 'label_smoothing':
            return nn.CrossEntropyLoss(label_smoothing=0.1)
        elif name == 'multi_margin':
            return nn.MultiMarginLoss()
        raise ValueError('Unknown loss')

    def fit(self, train_dataset):
        nn.Module.train(self, True)
        optimizer = self._build_optimizer()
        loss_function = self._build_loss()
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')
        loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.max_epoch):
            total_loss = 0.0
            all_pred, all_true = [], []

            for X_batch, y_batch in loader:
                y_pred = self.forward(X_batch)
                train_loss = loss_function(y_pred, y_batch)

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                # weight by batch size so the last (smaller) batch is averaged correctly
                total_loss += train_loss.item() * X_batch.size(0)
                all_pred.append(y_pred.argmax(dim=1).detach())
                all_true.append(y_batch)

            # accuracy is accumulated over the epoch while weights are changing,
            # so it slightly underestimates end-of-epoch accuracy
            epoch_loss = total_loss / len(train_dataset)
            accuracy_evaluator.data = {
                'true_y': torch.cat(all_true).numpy(),
                'pred_y': torch.cat(all_pred).numpy()
            }
            acc = accuracy_evaluator.evaluate()['accuracy']

            if self.history:
                self.history.record(Epoch=epoch, Loss=epoch_loss, Accuracy=acc)

            print('Epoch:', epoch, 'Accuracy:', acc, 'Loss:', epoch_loss)

        if self.history:
            self.history.save()

    def test(self, test_dataset):
        nn.Module.train(self, False)
        loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        preds = []
        with torch.no_grad():
            for X_batch, _ in loader:
                preds.append(self.forward(X_batch).argmax(dim=1))
        return torch.cat(preds)

    def run(self):
        print('method running...')
        print('--start training...')
        self.fit(self.data['train'])
        print('--start testing...')
        pred_y = self.test(self.data['test'])
        true_y = self.data['test'].tensors[1]
        return {'pred_y': pred_y, 'true_y': true_y}