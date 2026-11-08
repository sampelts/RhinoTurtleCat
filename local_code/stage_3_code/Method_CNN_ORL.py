'''
CNN for ORL faces (1x112x92 input, 40 classes)
'''

from local_code.stage_3_code.Method_CNN_Base import Method_CNN_Base
from torch import nn


class Method_CNN_ORL(Method_CNN_Base):
    max_epoch = 40
    batch_size = 32
    learning_rate = 5e-4
    use_dropout = True

    def __init__(self, mName, mDescription):
        super().__init__(mName, mDescription)

        # 1x112x92 -> 16x112x92 -> 16x56x46
        self.conv_1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn_1 = nn.BatchNorm2d(16)
        # 16x56x46 -> 32x56x46 -> 32x28x23
        self.conv_2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn_2 = nn.BatchNorm2d(32)
        # 32x28x23 -> 64x28x23 -> 64x14x11
        self.conv_3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn_3 = nn.BatchNorm2d(64)
        # 64x14x11 -> 64x14x11 -> 64x7x5
        self.conv_4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn_4 = nn.BatchNorm2d(64)

        self.activation_func = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.5)

        # 64*7*5 = 2240 -> 128 -> 40
        self.fc_layer_1 = nn.Linear(64 * 7 * 5, 128)
        self.fc_layer_2 = nn.Linear(128, 40)
        # no softmax: CrossEntropyLoss expects raw logits

    def forward(self, x):
        '''x: (batch, 1, 112, 92)'''
        h = self.pool(self.activation_func(self.bn_1(self.conv_1(x))))
        h = self.pool(self.activation_func(self.bn_2(self.conv_2(h))))
        h = self.pool(self.activation_func(self.bn_3(self.conv_3(h))))
        h = self.pool(self.activation_func(self.bn_4(self.conv_4(h))))
        h = h.flatten(start_dim=1)
        if self.use_dropout:
            h = self.dropout(self.activation_func(self.fc_layer_1(h)))
        else:
            h = self.activation_func(self.fc_layer_1(h))
        return self.fc_layer_2(h)

class Method_CNN_ORL_NoBatchNorm(Method_CNN_Base):
    max_epoch = 40
    batch_size = 32
    learning_rate = 5e-4
    use_dropout = True

    def __init__(self, mName, mDescription):
        super().__init__(mName, mDescription)

        # 1x112x92 -> 16x112x92 -> 16x56x46
        self.conv_1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        # 16x56x46 -> 32x56x46 -> 32x28x23
        self.conv_2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # 32x28x23 -> 64x28x23 -> 64x14x11
        self.conv_3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # 64x14x11 -> 64x14x11 -> 64x7x5
        self.conv_4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.activation_func = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.5)

        # 64*7*5 = 2240 -> 128 -> 40
        self.fc_layer_1 = nn.Linear(64 * 7 * 5, 128)
        self.fc_layer_2 = nn.Linear(128, 40)
        # no softmax: CrossEntropyLoss expects raw logits

    def forward(self, x):
        '''x: (batch, 1, 112, 92)'''
        h = self.pool(self.activation_func(self.conv_1(x)))
        h = self.pool(self.activation_func(self.conv_2(h)))
        h = self.pool(self.activation_func(self.conv_3(h)))
        h = self.pool(self.activation_func(self.conv_4(h)))
        h = h.flatten(start_dim=1)
        if self.use_dropout:
            h = self.dropout(self.activation_func(self.fc_layer_1(h)))
        else:
            h = self.activation_func(self.fc_layer_1(h))
        return self.fc_layer_2(h)
