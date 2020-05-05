import torch
import torch.nn as nn
import torch.nn.functional as F
from basic_functions import *

n = 2

class ConvNet(nn.Module):
    """
    A simple convolutional neural network shared among all pieces.
    """

    def __init__(self, input_chan, linear_dim, block_height, block_width):
        super().__init__()
        self.block_height = block_height
        self.block_width = block_width
        self.out_chans = [16, 32, 64, 128]
        self.linear_dim = linear_dim

        self.conv1 = nn.Conv2d(input_chan, self.out_chans[0], 3, 1, 1)
        self.conv1_bn = nn.BatchNorm2d(self.out_chans[0])
        self.conv2 = nn.Conv2d(self.out_chans[0], self.out_chans[1], 3, 1, 1)
        self.conv2_bn = nn.BatchNorm2d(self.out_chans[1])
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(self.out_chans[1], self.out_chans[2], 3, 1, 1)
        self.conv3_bn = nn.BatchNorm2d(self.out_chans[2])
        self.conv4 = nn.Conv2d(self.out_chans[2], self.out_chans[3], 3, 1, 1)
        self.conv4_bn = nn.BatchNorm2d(self.out_chans[3])
        self.pool2 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(self.out_chans[3] * (self.block_height // 4) * (self.block_width // 4), self.linear_dim)
        self.fc1_bn = nn.BatchNorm1d(self.linear_dim)
        # 128-d features
        self.fc2 = nn.Linear(self.linear_dim, self.linear_dim)
        self.fc2_bn = nn.BatchNorm1d(self.linear_dim)

    def forward(self, x):
        '''
        :param x: (N x C x block_height x block_width)
        :return:
        '''

        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = self.pool1(x)
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.relu(self.conv4_bn(self.conv4(x)))
        x = self.pool2(x)

        x = x.view(-1, self.out_chans[3] * (self.block_height // 4) * (self.block_width // 4))

        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = F.relu(self.fc2_bn(self.fc2(x)))

        return x


class JigsawNet(nn.Module):
    """
    A neural network that solves nxn jigsaw puzzles.
    """

    def __init__(self, input_chan, height, width, sinkhorn_iter=0):
        super().__init__()
        self.block_height = height // n
        self.block_width = width // n
        self.perm_inds = []  # perm_inds: left top coordinates of all blocks in the original image
        for r in range(0, height, self.block_height):
            for c in range(0, width, self.block_width):
                self.perm_inds.append((r, c))

        self.conv_linear_dim = 128
        self.linear_dim = 256


        self.conv_net = ConvNet(input_chan, self.conv_linear_dim, self.block_height, self.block_width)
        self.fc1 = nn.Linear(self.conv_linear_dim * n * n, self.linear_dim)
        self.fc1_bn = nn.BatchNorm1d(self.linear_dim)
        self.fc2 = nn.Linear(self.linear_dim, n**4)
        self.sinkhorn_iter = sinkhorn_iter

    def forward(self, x):
        '''
        :param x: (N x C x H x W)
        :return:
        '''

        x = torch.cat([self.conv_net(x[:, :, r:r + self.block_height, c:c + self.block_width])
                       for (r,c) in self.perm_inds], dim=1)
        x = F.dropout(x, p=0.1, training=self.training)
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = torch.sigmoid(self.fc2(x))

        if self.sinkhorn_iter > 0:
            x = x.view(-1, n*n, n*n)
            x = sinkhorn(x, self.sinkhorn_iter)
            x = x.view(-1, n**4)

        return x