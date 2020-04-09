import torch
import numpy as np
import matplotlib.pyplot as plt

def preprocess(images, n):
    batch_size, in_channel, height, width = images.size()

    # check if images can be fully divided into n*n pieces with equal length and width for each puzzle piece
    assert (n > 0 and height % n == 0 and width % n == 0)
    p_height = height // n
    p_width = width // n

    perm_inds = []
    for x in range(0, height, p_height):
        for y in range(0, width, p_width):
            perm_inds.append((x, y))

    return perm_inds, p_height, p_width


def permute_nxn(images, n):
    perm_inds, p_height, p_width = preprocess(images, n)

    batch_size, in_channel, height, width = images.size()

    # intialize shuffled image FloatTensor, shape = (batch_size, in_channel, height, width)
    p_images = torch.FloatTensor(images.size())
    # initialize permutation LongTensor, shape = (batch_size, n*n)
    perms = torch.LongTensor(batch_size, n * n)

    for i in range(batch_size):
        order = torch.randperm(n * n)  # jth piece is placed at order[j]th place
        for j in range(n * n):
            sr, sc = perm_inds[j]
            tr, tc = perm_inds[order[j]]
            p_images[i, :, tr:tr + p_height, tc:tc + p_width] = images[i, :, sr:sr + p_height, sc:sc + p_width]
        perms[i, :] = order

    return (p_images, perms)


def restore_nxn(p_images, perms, n):
    perm_inds, p_height, p_width = preprocess(p_images, n)

    batch_size, in_channel, height, width = p_images.size()

    images = torch.FloatTensor(p_images.size())

    for i in range(batch_size):
        for j in range(n * n):
            sr, sc = perm_inds[j]
            tr, tc = perm_inds[perms[i, j]]
            images[i, :, sr:sr + p_height, sc:sc + p_height] = p_images[i, :, tr:tr + p_width, tc:tc + p_width]

    return images


def perm2vecmat(perms, n):
    batch_size, n_pieces = perms.size()
    mat = torch.zeros(batch_size, n * n, n * n)

    # m[i][j] : i is assigned to j
    for i in range(batch_size):
        for p in range(n * n):
            mat[i, p, perms[i, p]] = 1.
    return mat.view(batch_size, -1)


def vecmat2perm(x, n):
    batch_size = x.size()[0]
    x = x.view(batch_size, n * n, n * n)
    _, ind = x.max(2)
    return ind



def sinkhorn(A, n_iter=4):
    """
    Sinkhorn iterations.
    """
    for i in range(n_iter):
        A = A / A.sum(dim=1, keepdim=True)
        A = A / A.sum(dim=2, keepdim=True)
    return A




def imshow(img, title=None):
    """
    Displays a torch image.
    """
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if title != None:
        plt.title(title)