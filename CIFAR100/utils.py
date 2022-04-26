import errno
import os
import os.path as osp
import shutil
import random
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm


class AverageMeter(object):
    """Computes and stores the average and current value"""

    # 滑动平均
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    # top k准确率
    # 预测结果前k个中出现的正确结果的次数
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def mkdir_if_missing(directory):
    # 创建文件夹，如果这个文件夹不存在的话
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def save_checkpoint(state, is_best=False, fpath=''):
    if len(osp.dirname(fpath)) != 0:
        mkdir_if_missing(osp.dirname(fpath))
    torch.save(state, fpath)
    if is_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), 'best_model.pth.tar'))


def adjust_learning_rate(optimizer, epoch, initial_lr):
    lr = initial_lr
    if epoch >= 60:
        lr /= 10
    if epoch >= 80:
        lr /= 10
    if epoch >= 100:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warp_tqdm(data_loader, disable_tqdm):
    # 进度条打印
    if disable_tqdm:
        tqdm_loader = data_loader
    else:
        tqdm_loader = tqdm(data_loader, ncols=0)
    return tqdm_loader


def load_checkpoint(model, checkpoint_PATH, optimizer):
    model_CKPT = torch.load(checkpoint_PATH)
    model.load_state_dict(model_CKPT['state_dict'])
    print('loading checkpoint!')
    optimizer.load_state_dict(model_CKPT['optimizer'])
    return model, optimizer


def train(train_loader, model, criterion, optimizer, epoch, writer=None):
    # 每个epoch的优化过程
    losses = AverageMeter()

    # switch to train mode
    model.train()

    for input, target in warp_tqdm(train_loader, True):
        input = input.cuda()
        target = target.cuda()

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

    if writer is not None:
        writer.add_scalar('Loss', losses.avg, epoch + 1)
        writer.add_figure('predictions vs actuals', plot_classes_preds(model, input, target), global_step=epoch + 1)

    log = 'Epoch:{0}\tLoss: {loss.avg:.4f}\t'.format(epoch + 1, loss=losses)
    return log


def test(test_loader, model, criterion, epoch, writer=None):
    top1 = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for input, target in test_loader:
        # compute output
        with torch.no_grad():
            input = input.cuda()
            target = target.cuda()
            output = model(input)

        # measure accuracy and record loss
        acc1 = accuracy(output.data, target)[0]
        loss = criterion(output, target)
        losses.update(loss.item(), input.size(0))
        top1.update(acc1.item(), input.size(0))

    if writer is not None:
        writer.add_scalar('Loss', losses.avg, epoch + 1)
        writer.add_scalar('Test Accuarcy', top1.avg, epoch + 1)

    log = 'Test Acc@1: {top1.avg:.3f}\tLoss: {loss.avg:.4f}'.format(top1=top1, loss=losses)

    return top1.avg, log


def init_seeds(seed=0):
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)  # seed for PyTorch CPU
    torch.cuda.manual_seed(seed)  # seed for current PyTorch GPU
    torch.cuda.manual_seed_all(seed)  # seed for all PyTorch GPUs
    if seed == 0:
        # if True, causes cuDNN to only use deterministic convolution algorithms.
        torch.backends.cudnn.deterministic = True
        # if True, causes cuDNN to benchmark multiple convolution algorithms and select the fastest.
        torch.backends.cudnn.benchmark = False


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_mixup(train_loader, model, criterion, optimizer, alpha, epoch, writer=None):
    losses = AverageMeter()

    # switch to train mode
    model.train()

    for input, target in warp_tqdm(train_loader, True):
        input = input.cuda()
        target = target.cuda()

        input, target_a, target_b, lam = mixup_data(input, target, alpha, True)

        # compute output
        output = model(input)
        loss = mixup_criterion(criterion, output, target_a, target_b, lam)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

    if writer is not None:
        writer.add_scalar('Loss', losses.avg, epoch + 1)

    log = 'Epoch:{0}\tLoss: {loss.avg:.4f}\t'.format(epoch + 1, loss=losses)
    return log


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    # 限制坐标区域不超过样本大小
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def train_cutmix(train_loader, model, criterion, optimizer, beta, cutmix_prob, epoch, writer=None):
    losses = AverageMeter()

    # switch to train mode
    model.train()

    for input, target in warp_tqdm(train_loader, True):
        input = input.cuda()
        target = target.cuda()

        # cutmix
        r = np.random.rand(1)
        if beta > 0 and r < cutmix_prob:
            lam = np.random.beta(beta, beta)
            rand_index = torch.randperm(input.size()[0]).cuda()
            target_a = target
            target_b = target[rand_index]

            bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
            input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))

            output = model(input)
            loss = mixup_criterion(criterion, output, target_a, target_b, lam)
        else:
            output = model(input)
            loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

    if writer is not None:
        writer.add_scalar('Loss', losses.avg, epoch + 1)

    log = 'Epoch:{0}\tLoss: {loss.avg:.4f}\t'.format(epoch + 1, loss=losses)
    return log


# for tensorboard
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    npimg = img.cpu().numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def images_to_probs(net, images):
    """
   Generates predictions and corresponding probabilities from a trained
   network and a list of images
   """
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.cpu().numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]  # 返回预测结果及概率


def plot_classes_preds(net, images, labels):
    """
   Generates matplotlib Figure using a trained network, along with images
   and labels from a batch, that shows the network's top prediction along
   with its probability, alongside the actual label, coloring this
   information based on whether the prediction was correct or not.
   Uses the "images_to_probs" function.
   """
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(10, 10))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx + 1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            preds[idx],
            probs[idx] * 100.0,
            labels[idx]),
            color=("green" if preds[idx] == labels[idx].item() else "red"))
    return fig
