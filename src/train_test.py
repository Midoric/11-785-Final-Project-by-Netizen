import torch
from basic_functions import *

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
n = 2

def train(epoch, train_loader, model, criterion, optimizer):
    model.train()
    total_loss = 0.
    n_samples = 0
    n_correct_pred = 0
    for i, (images, _) in enumerate(train_loader):
        '''
        images: N x C x H x W
        '''
        p_images, perms = permute_nxn(images, n)
        label_mat = perm2vecmat(perms, n)
        p_images, label_mat, perms = p_images.to(DEVICE), label_mat.to(DEVICE), perms.to(DEVICE)
        pred_mat = model(p_images)


        loss = criterion(pred_mat, label_mat)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        n_samples += images.shape[0]
        n_correct_pred += compute_acc(vecmat2perm(pred_mat, n), perms, n, False).item()
        total_loss += loss.item() * images.shape[0]
    
    print("Epoch {}: train_loss = {}, train_acc = {}".format(epoch+1, total_loss / n_samples, n_correct_pred / n_samples))
    return total_loss / n_samples, n_correct_pred / n_samples

def val(epoch, val_loader, model, criterion, optimizer):
    model.eval()
    total_loss = 0.
    n_samples = 0
    n_correct_pred = 0
    for i, (images, _) in enumerate(val_loader):
        p_images, perms = permute_nxn(images, n)
        label_mat = perm2vecmat(perms, n)
        p_images, label_mat, perms = p_images.to(DEVICE), label_mat.to(DEVICE), perms.to(DEVICE)
        pred_mat = model(p_images)

        loss = criterion(pred_mat, label_mat)

        n_samples += images.shape[0]
        n_correct_pred += compute_acc(vecmat2perm(pred_mat, n), perms, n, False).item()
        total_loss += loss.item() * images.shape[0]

    print("Epoch {}: val_loss = {}, val_acc = {}".format(epoch+1, total_loss / n_samples, n_correct_pred / n_samples))
    return total_loss / n_samples, n_correct_pred / n_samples


# Test process
def test(epoch, test_loader, model, criterion, optimizer):
    model.eval()
    total_acc = 0.
    n_samples = 0
    for i, (images, _) in enumerate(test_loader):
        p_images, perms = permute_nxn(images, n)
        p_mat = perm2vecmat(perms, n)

        p_images, p_mat, perms = p_images.to(DEVICE), p_mat.to(DEVICE), perms.to(DEVICE)
        pred_mat = model(p_images)
        
        n_samples += images.shape[0]
        total_acc += compute_acc(vecmat2perm(pred_mat, n), perms, False).item()

    print("Epoch {}: test acc = {}".format(epoch, total_acc / n_samples))
    return total_acc / n_samples


def compute_acc(p_pred, p_true, n, average=True):
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

