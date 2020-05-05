import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

dataset_dir = './data'

transform = transforms.Compose(
    [transforms.ToTensor()])

def load_data(batch_size, num_workers=0, val_ratio=0.2):

    train_val_set = torchvision.datasets.CIFAR100(root=dataset_dir, train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR100(root=dataset_dir, train=False, download=True, transform=transform)

    total = len(train_val_set)
    ind = list(range(total))
    n_train = int(np.floor((1. - val_ratio) * total))
    train_ind, val_ind = ind[:n_train], ind[n_train:]
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_ind)
    val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_ind)

    train_loader = torch.utils.data.DataLoader(train_val_set, batch_size=batch_size, shuffle=False,
                                               sampler = train_sampler, num_workers=num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(train_val_set, batch_size=batch_size, shuffle=False,
                                             sampler = val_sampler, num_workers=num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                              num_workers=num_workers)

    return train_loader, val_loader, test_loader
