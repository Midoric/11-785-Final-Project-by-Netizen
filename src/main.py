'''
CMU 11-785 Final Project (Final version)
Team: Netizen
Members: Mingyi Cai, Yi Ren, Daijun Qian, Min Hsuan Hsu
Credits to https://research.wmz.ninja/attachments/articles/2018/03/jigsaw_cifar100.html
'''

from train_test import *
from model import *
from dataloader import *
from basic_functions import *
import matplotlib.pyplot as plt
import torch.optim as optim

torch.manual_seed(11785)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATH = ''


def main():
    sinkhorn_iter = 5

    batch_size = 32 if DEVICE == 'cuda' else 1
    num_workers = 16
    val_ratio = 0.2
    train_loader, val_loader, test_loader = load_data(batch_size=batch_size, num_workers=num_workers,
                                                      val_ratio=val_ratio)

    input_chan, height, width = next(iter(train_loader))[0].shape[1:]

    n_epochs = 50
    model = JigsawNet(input_chan=input_chan, height=height, width=width, sinkhorn_iter=sinkhorn_iter).to(DEVICE)
    model.apply(init_weights)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience = 0)

    print("Model Architecture:", model)
    n_params = 0
    for p in model.parameters():
        n_params += np.prod(p.size())
    print('# of parameters: {}'.format(n_params))


    torch.cuda.empty_cache()
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []
    for epoch in range(n_epochs):
        print("Epoch {}: Learning rate = {}".format(epoch + 1, get_lr(optimizer)))
        train_loss, train_acc = train(epoch, train_loader, model, criterion, optimizer)
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)

        val_loss, val_acc = val(epoch, val_loader, model, criterion)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)

        scheduler.step(val_loss)
        # test_acc = test(epoch, test_loader, model)
        # torch.save(model.state_dict(), 'jigsaw_cifar100_e{}_s{}.pt'.format(epoch, sinkhorn_iter))

    plt.figure()
    plt.plot(train_loss_history)
    plt.plot(val_loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'])
    plt.savefig(PATH + 'loss')

    plt.figure()
    plt.plot(train_acc_history)
    plt.plot(val_acc_history)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'])
    plt.savefig(PATH + 'acc')


if __name__ == '__main__':
    main()




