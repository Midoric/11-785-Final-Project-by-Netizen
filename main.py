'''
CMU 11-785 Final Project (Midterm report version)
Team: Netizen
Partly cited from https://research.wmz.ninja/attachments/articles/2018/03/jigsaw_cifar100.html
'''

import torch
import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from basic_functions import *
from model import *


torch.manual_seed(11785)

device = "cude" if torch.cuda.is_available() else "cpu"
dataset_dir = './data'

n = 3
batch_size = 32
n_epochs = 20
sinkhorn_iter = 5

model = JigsawNet(sinkhorn_iter=sinkhorn_iter).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

n_params = 0
for p in model.parameters():
    n_params += np.prod(p.size())
print('# of parameters: {}'.format(n_params))




transform = transforms.Compose(
    [transforms.Resize(36)])

train_set = torchvision.datasets.CIFAR10(root=dataset_dir, train=True, download=True, transform=transform)
dev_set = torchvision.datasets.CIFAR10(root=dataset_dir, train=True, download=True, transform=transform)
test_set = torchvision.datasets.CIFAR100(root=dataset_dir, train=False, download=True, transform=transform)


# Prepare training, validation, and test samples.
validation_ratio = 0.2
total = len(train_set)
ind = list(range(total))
n_train = int(np.floor((1. - validation_ratio) * total))
train_ind, validation_ind = ind[:n_train], ind[n_train:]
train_subsampler = torch.utils.data.sampler.SubsetRandomSampler(train_ind)
validation_subsampler = torch.utils.data.sampler.SubsetRandomSampler(validation_ind)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=256,
                                           sampler=train_subsampler, num_workers=16, pin_memory=True)
validation_loader = torch.utils.data.DataLoader(dev_set, batch_size=256,
                                        sampler=validation_subsampler, num_workers=16, pin_memory=True)

test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False, num_workers=16)



# Training process
def train_model(train_loader, validation_loader):
    for epoch in range(n_epochs):
        # Training phase
        model.train()
        running_loss = 0.
        n_correct_pred = 0
        n_samples = 0
        for i, data in enumerate(train_loader, 0):
            inputs, _ = data
            x_in, perms = permute_nxn(inputs, n)
            y_in = perm2vecmat(perms, n)
            n_samples += inputs.size()[0]
            x_in, y_in, perms = x_in.to(device), y_in.to(device), perms.to(device)

            optimizer.zero_grad()
            outputs = model(x_in)
            n_correct_pred += compute_acc(vecmat2perm(outputs), perms, False).item()
            loss = criterion(outputs, y_in)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x_in.size()[0]

        print("train loss : {.4f}, train acc : {.4f}".format(running_loss/n_samples, n_correct_pred/n_samples))

        # Validation phase
        model.eval()
        running_loss = 0.
        n_correct_pred = 0
        n_samples = 0
        for i, data in enumerate(validation_loader, 0):
            inputs, _ = data
            x_in, perms = permute_nxn(inputs, n)
            y_in = perm2vecmat(perms, n)
            n_samples += inputs.size()[0]
            x_in, y_in, perms = x_in.to(device), y_in.to(device), perms.to(device)

            outputs = model(x_in)
            n_correct_pred += compute_acc(vecmat2perm(outputs, n), perms, False).item()
            loss = criterion(outputs, y_in)
            running_loss += loss.item() * x_in.size()[0]

        torch.save(model.state_dict(), 'jigsaw_cifar100_e{}_s{}.pt'.format(epoch, sinkhorn_iter))

        print("val loss : {.4f}, val acc : {.4f}".format(running_loss/n_samples, n_correct_pred/n_samples))
    print('Training completed')



# Test process
def test_model(test_loader):
    running_acc = 0.
    n = 0
    model.eval()
    for i, data in enumerate(test_loader, 0):
        inputs, _ = data
        x_in, perms = permute_nxn(inputs, n)

        y_in = perm2vecmat(perms, n)
        x_in, y_in, perms = x_in.to(device), y_in.to(device), perms.to(device)
        pred = model(x_in)
        perms_pred = vecmat2perm(pred.cpu().data, n)
        running_acc += compute_acc(perms_pred, perms, False)
        n += x_in.shape[0]
    acc = running_acc / n
    return acc



# Test helper
def compute_acc(p_pred, p_true, average=True):
    """
    We require that the location of all four pieces are correctly predicted.
    Note: this function is compatible with GPU tensors.
    """
    # Remember to cast to float.
    n = torch.sum((torch.sum(p_pred == p_true, 1) == n**2).float())
    if average:
        return n / p_pred.size()[0]
    else:
        return n



train_model(train_loader, validation_loader)
test_model(test_loader)
