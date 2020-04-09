import torch
import torch.nn as nn
import torch.nn.functional as F
from basic_functions import *

class SimpleConvNet(nn.Module):
    """
    A simple convolutional neural network shared among all pieces.
    """

    def __init__(self):
        super().__init__()
        # 3 x 32 x 32 input
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.conv1_bn = nn.BatchNorm2d(16)
        # 8 x 32 x 32
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.conv2_bn = nn.BatchNorm2d(32)
        # 16 x 32 x 32
        self.pool1 = nn.MaxPool2d(2, 2)
        # 16 x 16 x 16
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv3_bn = nn.BatchNorm2d(64)
        # 32 x 16 x 16
        self.conv4 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv4_bn = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc1_bn = nn.BatchNorm1d(128)
        # 128-d features
        self.fc2 = nn.Linear(128, 128)
        self.fc2_bn = nn.BatchNorm1d(128)

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = self.pool1(x)
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.relu(self.conv4_bn(self.conv4(x)))
        x = self.pool2(x)

        x = x.view(-1, 128 * 3 * 3)
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = F.relu(self.fc2_bn(self.fc2(x)))

        return x


class JigsawNet(nn.Module):
    """
    A neural network that solves 2x2 jigsaw puzzles.
    """

    def __init__(self, sinkhorn_iter=0):
        super().__init__()
        self.conv_net = SimpleConvNet()
        self.fc1 = nn.Linear(128 * 9, 256)
        self.fc1_bn = nn.BatchNorm1d(256)
        # 4 x 4 assigment matrix
        self.fc2 = nn.Linear(256, 81)
        self.sinkhorn_iter = sinkhorn_iter

    def forward(self, x):
        # Split the input into four pieces and pass them into the
        # same convolutional neural network.

        # Cat

        x = torch.cat([self.conv_net(x[:, :, i:i + 12, j:j + 12]) for i in range(0, 36, 12) for j in range(0, 36, 12)],
                      dim=1)
        # Dense layer

        # x = F.dropout(x, p=0.1, training=self.training)
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = torch.sigmoid(self.fc2(x))
        if self.sinkhorn_iter > 0:
            x = x.view(-1, 9, 9)
            x = sinkhorn(x, self.sinkhorn_iter)
            x = x.view(-1, 81)

        return x
