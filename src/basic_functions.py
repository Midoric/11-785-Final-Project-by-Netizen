import torch
import numpy as np
import matplotlib.pyplot as plt

def preprocess(images, n):
    '''
    :param images: (N x C x H x W) input images
    :param n: int such the image is divided into n x n blocks
    :return perm_inds: left top coordinates of all blocks in the original image
    :return block_height: height of each block
    :return block_width: width of each block
    '''
    batch_size, channel, height, width = images.size()
    # check if images can be fully divided into n x n pieces with equal length and width for each puzzle piece
    assert (n > 0 and height % n == 0 and width % n == 0)

    block_height, block_width = height // n, width // n

    perm_inds = []
    for r in range(0, height, block_height):
        for c in range(0, width, block_width):
            perm_inds.append((r, c))

    return perm_inds, block_height, block_width

def permute_nxn(images, n):
    '''
    :param images: (N x C x H x W) input images
    :param n: int such the image is divided into n x n blocks
    :return permuted_images: (N x C x H x W) permuted images
    :return perms: (N x (n**2)) all permutations
    '''
    perm_inds, block_height, block_width = preprocess(images, n)
    batch_size, in_channel, height, width = images.size()

    permuted_images = torch.FloatTensor(images.size()) # Intialized as FloatTensor, dim: (N x C x H x W)
    perms = torch.LongTensor(batch_size, n * n)    # Initialized as permutation LongTensor, dim: (N, n x n)

    for i in range(batch_size):
        order = torch.randperm(n * n)  # jth piece is placed at order[j]th place
        for j in range(n * n):
            sr, sc = perm_inds[j]
            tr, tc = perm_inds[order[j]]
            permuted_images[i, :, tr:tr + block_height, tc:tc + block_width] = images[i, :, sr:sr + block_height, sc:sc + block_width]
        perms[i, :] = order

    return (permuted_images, perms)

# def restore_nxn(p_images, perms, n):
#     perm_inds, p_height, p_width = preprocess(p_images, n)
#     batch_size, in_channel, height, width = p_images.size()
#
#     images = torch.FloatTensor(p_images.size())
#
#     for i in range(batch_size):
#         for j in range(n * n):
#             sr, sc = perm_inds[j]
#             tr, tc = perm_inds[perms[i, j]]
#             images[i, :, sr:sr + p_height, sc:sc + p_height] = p_images[i, :, tr:tr + p_width, tc:tc + p_width]
#
#     return images

def perm2vecmat(perms, n):
    '''
    :param perms: (N x (n**2)) all permutations
    :param n: int such the image is divided into n x n blocks
    :return M: (N x (n**4)) all permutation matrices
    '''

    batch_size = perms.size()[0]
    M = torch.zeros(batch_size, n * n, n * n)

    # m[i][j] : i is assigned to j
    for i in range(batch_size):
        for p in range(n * n):
            M[i, p, perms[i, p]] = 1.
    return M.view(batch_size, -1)

def vecmat2perm(M, n):
    '''
    :param M: (N x (n**4)) permuration matrices
    :param n: n: int such the image is divided into n x n blocks
    :return perms: (N x (n**2)) all permutations
    '''

    batch_size = M.size()[0]
    M = M.view(batch_size, n * n, n * n)
    _, perms = M.max(2)
    return perms

def sinkhorn(A, n_iter=4):
    """
    Sinkhorn iterations.
    """
    for i in range(n_iter):
        A = A / A.sum(dim=1, keepdim=True)
        A = A / A.sum(dim=2, keepdim=True)
    return A

