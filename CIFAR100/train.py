import warnings

import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from torchvision.models.resnet import resnet18
from torchtoolbox.transform import Cutout

from utils import *

warnings.filterwarnings('ignore')

# load data
cifar_data_path = 'data/cifar100'
mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
normalize = T.Normalize(mean=mean, std=std)
cifar_train_transform = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    Cutout(),
    T.ToTensor(),
    normalize,
])
cifar_test_transform = T.Compose([
    T.ToTensor(),
    normalize,
])
cifar_train = CIFAR100(cifar_data_path, train=True, transform=cifar_train_transform)
cifar_test = CIFAR100(cifar_data_path, train=False, transform=cifar_test_transform)
cifar_train_loader = DataLoader(cifar_train, batch_size=256, shuffle=True, num_workers=2, pin_memory=False)
cifar_test_loader = DataLoader(cifar_test, batch_size=1024, num_workers=2, pin_memory=False)


if __name__ == '__main__':
    criterion = nn.CrossEntropyLoss()
    model = resnet18(pretrained=False, num_classes=100).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.001)

    # load model
    # model_CKPT = torch.load('init.pth.tar')
    # model.load_state_dict(model_CKPT['state_dict'])
    # optimizer.load_state_dict(model_CKPT['optimizer'])
    # best_acc = model_CKPT['acc']
    # init_epoch = model_CKPT['epoch']
    best_acc = 0
    init_epoch = 0

    # start training
    print('start training')
    alpha = 1.0
    prob = 0.5
    num_epochs = 100
    train_loader = cifar_train_loader
    test_loader = cifar_test_loader
    epoch = 0
    for epoch in range(init_epoch, init_epoch+num_epochs):
        adjust_learning_rate(optimizer, epoch, 0.1)
        train_log = train(train_loader, model, criterion, optimizer, epoch)
        # train_log = train_mixup(train_loader, model, criterion, optimizer, alpha, epoch)
        # train_log = train_cutmix(train_loader, model, criterion, optimizer, alpha, prob, epoch)
        acc, test_log = test(test_loader, model, criterion)
        log = train_log + test_log
        print(log)
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        if is_best:
            save_checkpoint({'epoch': epoch + 1,
                             'state_dict': model.state_dict(),
                             'acc': acc,
                             }, False, 'best_model_cutout.pth.tar')
    torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'best_acc': best_acc,
                'optimizer': optimizer.state_dict()},
               'model.pth.tar')
