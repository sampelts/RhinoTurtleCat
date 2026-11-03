from local_code.stage_3_code.Method_CNN_Base import Method_CNN_Base
from torch import nn

class Method_CNN_MNIST(Method_CNN_Base):
    max_epoch = 20
    batch_size = 128
    learning_rate = 1e-3

    def __init__(self, mName, mDescription):
        super().__init__(mName, mDescription)

        # padding=2 makes 28x28 input behave like the 32x32 input LeNet-5 was designed for
        # 1x28x28 -> 6x28x28 -> 6x14x14
        self.conv_1 = nn.Conv2d(1, 6, kernel_size=5, padding= 2)
        self.activation_func_1 = nn.ReLU()
        self.pool_1 = nn.MaxPool2d(2)
        # 6x14x14 -> 16x10x10 -> 16x5x5
        self.conv_2 = nn.Conv2d(6, 16, kernel_size=5)
        self.activation_func_2 = nn.ReLU()
        self.pool_2 = nn.MaxPool2d(2)
        # 16*5*5 = 400 -> 120 -> 84 -> 10
        self.fc_layer_1 = nn.Linear(16 * 5 * 5, 120)
        self.activation_func_3 = nn.ReLU()
        self.fc_layer_2 = nn.Linear(120, 84)
        self.activation_func_4 = nn.ReLU()
        self.fc_layer_3 = nn.Linear(84, 10)
        # no softmax: CrossEntropyLoss expects raw logits

    def forward(self, x):
        h = self.pool_1(self.activation_func_1(self.conv_1(x)))
        h = self.pool_2(self.activation_func_2(self.conv_2(h)))
        h = h.flatten(start_dim=1)
        h = self.activation_func_3(self.fc_layer_1(h))
        h = self.activation_func_4(self.fc_layer_2(h))
        return self.fc_layer_3(h)