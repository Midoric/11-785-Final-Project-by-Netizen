import torch

def preprocess(images, n):
    '''
    :param images: (N x C x H x W) input images
    :param n: int such the image is divided into n x n blocks
    :return perm_inds: left top coordinates of all blocks in the original image
    :return block_height: height of each block
    :return block_width: width of each block
    '''
    batch_size, channel, h, w = images.size()
    assert (n > 0 and h % n == 0 and w % n == 0)

    # check if images can be fully divided into n x n pieces with equal length and width for each puzzle piece

    bh, bw = h // n, w // n                # height, width of each block

    perm_inds = []
    for r in range(0, h, bh):
        for c in range(0, w, bw):
            perm_inds.append((r, c))

    return perm_inds, bh, bw

def permute_nxn(images, n):
    '''
    :param images: (N x C x H x W) input images
    :param n: int such the image is divided into n x n blocks
    :return permuted_images: (N x C x H x W) permuted images
    :return perms: (N x (n**2)) all permutations
    '''
    perm_inds, bh, bw = preprocess(images, n)
    batch_size = images.size()[0]

    permuted_images = torch.FloatTensor(images.size())  # Initialized as FloatTensor, dim: (N x C x H x W)
    perms = torch.LongTensor(batch_size, n * n)         # Initialized as permutation LongTensor, dim: (N, n**2)

    for i in range(batch_size):
        order = torch.randperm(n * n)  # jth piece is placed at order[j]th place
        for j in range(n * n):
            sr, sc = perm_inds[j]
            tr, tc = perm_inds[order[j]]
            permuted_images[i, :, tr:tr + bh, tc:tc + bw] = images[i, :, sr:sr + bh, sc:sc + bw]
        perms[i, :] = order

    return permuted_images, perms

def restore_nxn(p_images, perms, n):
    '''
    :params p_images: (N x C x H x W) permuted images
    :param perms: permutations
    :param n: int such the image is divided into n x n blocks
    :return images (N x C x H x W) original images
    '''
    perm_inds, bh, bw = preprocess(p_images, n)
    batch_size = p_images.size()[0]

    images = torch.FloatTensor(p_images.size())

    for i in range(batch_size):
        for j in range(n * n):
            sr, sc = perm_inds[j]
            tr, tc = perm_inds[perms[i, j]]
            images[i, :, sr:sr + bh, sc:sc + bh] = p_images[i, :, tr:tr + bw, tc:tc + bw]

    return images

def perm2vecmat(perms, n):
    '''
    :param perms: (N x (n**2)) all permutations
    :param n: int such the image is divided into n x n blocks
    :return M: (N x (n**4)) all permutation matrices
    '''
    assert (perms.shape[1] == n ** 2)

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
    assert (M.shape[1] == n ** 4)

    batch_size = M.size()[0]
    M = M.view(batch_size, n * n, n * n)
    _, perms = M.max(2)
    return perms

def sinkhorn(M, n_iter=4):
    '''
    :param M: (N x (n*n) x (n*n)) original matrix
    :param n_iter: number of iterations to be performed
    :return M: (N x (n*n) x (n*n)) matrix after sinkhorn normalization
    '''
    for i in range(n_iter):
        M = M / M.sum(dim=1, keepdim=True)
        M = M / M.sum(dim=2, keepdim=True)
    return M

def get_lr(optimizer):
    '''
    Get learning rate of the optimizer
    '''
    for param_group in optimizer.param_groups:
        return param_group['lr']


