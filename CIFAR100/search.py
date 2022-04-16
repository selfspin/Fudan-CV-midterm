from itertools import product
from train import *

if __name__ == '__main__':
    lrs = [1e-1, 1e-2, 1e-3, 1e-4]
    wds = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
    moms = [0, 0.9]
    params_list = list(product(lrs, wds, moms))
    best_params = params_list[0]

    best_acc = 0
    for lr, wd, mom in params_list:
        print(lr, wd, mom)
        loc_acc = 0
        criterion = nn.CrossEntropyLoss()
        model = resnet18(pretrained=False, num_classes=100).cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=mom, weight_decay=wd)

        num_epochs = 120
        alpha = 1.0
        prob = 0.5
        train_loader = cifar_train_loader
        test_loader = cifar_test_loader

        for epoch in range(num_epochs):
            adjust_learning_rate(optimizer, epoch, 0.1)
            train_log = train(train_loader, model, criterion, optimizer, epoch)
            # train_log = train_mixup(train_loader, model, criterion, optimizer, alpha, epoch)
            # train_log = train_cutmix(train_loader, model, criterion, optimizer, alpha, prob, epoch)
            acc, test_log = test(test_loader, model, criterion)
            log = train_log + test_log
            print(log)
            is_best = acc > best_acc
            loc_acc = max(acc, loc_acc)
            best_acc = max(acc, best_acc)
            if is_best:
                save_checkpoint({'epoch': epoch + 1,
                                 'state_dict': model.state_dict(),
                                 'acc': acc,
                                 }, False, 'search_best_model_cutout.pth.tar')

        file_handle = open('record.txt', mode='a')
        file_handle.write('{0} {1} {2} {3}\n'.format(lr, wd, mom, loc_acc))
        file_handle.close()
