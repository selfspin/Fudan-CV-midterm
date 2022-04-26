# CIFAR100

## Requirements

TODO

## Model

resnet18 (w/o pretraining)

## Results

Top 1 accuracy(%)

- baseline:  61.74
- mixup: 62.40
- cutout: TODO
- cutmix: 64.66

## Training

- **baseline**

**optimizer:** SGD

**hyper-parameters:**

learning rate = 0.1

momentum = 0.9

weight decay = 0.001

学习率在第48、60、72 epoch 除以10，第100 epoch时停止训练



直接运行train.py进行训练

使用tensorboard，命令行输入下述命令可以监视loss曲线等情况

```
tensorboard --logdir=log
```

路径可在train.py第41-43行更改：

```python
writer = SummaryWriter('log')
writer_train = SummaryWriter('log/train')
writer_test = SummaryWriter('log/test')
```

<img src="CIFAR100\figures\loss_bl.png" alt="loss_bl" style="zoom: 50%;" /> <img src="CIFAR100\figures\acc_bl.png" alt="acc_bl" style="zoom:50%;" />



- **mixup**

**optimizer:** SGD

**hyper-parameters:**

learning rate = 0.1

momentum = 0.9

weight decay = 0.001

学习率在第60、80、100 epoch 除以10，第120 epoch时停止训练



在train.py中修改76-78行：

```python
train_log = train(train_loader, model, criterion, optimizer, epoch, writer_train)
# train_log = train_mixup(train_loader, model, criterion, optimizer, alpha, epoch, writer_train)
# train_log = train_cutmix(train_loader, model, criterion, optimizer, alpha, prob, epoch, writer_train)
```

注释上述第1行，取消注释第2行，然后运行train.py即可对mixup增强数据训练

tensorboard可视化：

<img src="CIFAR100\figures\loss_mu.png" alt="loss_mu" style="zoom:50%;" /> <img src="CIFAR100\figures\acc_mu.png" alt="acc_mu" style="zoom:50%;" /> 

  

- **cutout**

**optimizer:** SGD

**hyper-parameters:**

learning rate = 0.1

momentum = 0.9

weight decay = 0.001

学习率在第60、80、100 epoch 除以10，第120 epoch时停止训练



cutout调用torchtoolbox包，因此直接在transform中加入Cutout()：

在train.py中修改22-28行：

```python
cifar_train_transform = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    # Cutout(),
    T.ToTensor(),
    normalize,
])
```

取消注释Cutout()即可训练，不需要修改76-78行的train函数



- cutmix

**optimizer:** SGD

**hyper-parameters:**

learning rate = 0.1

momentum = 0

weight decay = 0.005

学习率在第60、80、100 epoch 除以10，第120 epoch时停止训练



在train.py中修改76-78行：

```python
train_log = train(train_loader, model, criterion, optimizer, epoch, writer_train)
# train_log = train_mixup(train_loader, model, criterion, optimizer, alpha, epoch, writer_train)
# train_log = train_cutmix(train_loader, model, criterion, optimizer, alpha, prob, epoch, writer_train)
```

注释上述第1行，取消注释第3行，然后运行train.py即可对cutmix增强数据训练

tensorboard可视化：

<img src="CIFAR100\figures\loss_cm.png" alt="loss_cm" style="zoom:50%;" /> <img src="CIFAR100\figures\acc_cm.png" alt="acc_cm" style="zoom:50%;" /> 



## 更多可视化结果

经归一化和标准数据增强（crop/flip）的部分图像输入

<img src="CIFAR100\figures\inputs.png" alt="inputs" style="zoom:80%;" /> 



对部分特定类别的预测准确率，图中标签82（sunflower）、30（dolphin）、36（hamster）预测非常准确，而对35（girl）准确率较低

<img src="CIFAR100\figures\pva.png" alt="pva" style="zoom:80%;" />  

