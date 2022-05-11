# CIFAR100

使用resnet18在CIFAR100数据集上训练，并使用了mixup, cutout, cutmix三种数据增强方法分别训练

### Requirements

TODO: 版本

pytorch

torchtoolbox

numpy

matplotlib

下面这些不知道要不要，哪些是python自带的

tqdm

warnings

errno

os

shutil

itertools

random

### Results

Top 1 accuracy(%)

- baseline:  61.74
- mixup: 62.40
- cutout: 61.48
- cutmix: 64.66

### Training

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

<img src="figures\loss_bl.png" alt="loss_bl" style="zoom: 67%;" /> <img src="\figures\acc_bl.png" alt="acc_bl" style="zoom:67%;" />



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

<img src="figures\loss_mu.png" alt="loss_mu" style="zoom:67%;" />  <img src="\figures\acc_mu.png" alt="acc_mu" style="zoom:67%;" /> 

  

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

tensorboard可视化：

<img src="figures\loss_co.png" alt="loss_co" style="zoom: 67%;" /> <img src="figures\acc_co.png" alt="acc_co" style="zoom: 67%;" /> 



- **cutmix**

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

<img src="figures\loss_cm.png" alt="loss_cm" style="zoom:67%;" /> <img src="\figures\acc_cm.png" alt="acc_cm" style="zoom:67%;" /> 



### 更多可视化结果

- mixup 图像

<img src="figures\apple+boy_mixup.png" alt="apple+boy_mixup" style="zoom: 33%;" /> <img src="figures\boy+elephant_mixup.png" alt="boy+elephant_mixup" style="zoom: 33%;" /> <img src="figures\apple+elephant_mixup.png" alt="apple+elephant_mixup" style="zoom: 33%;" /> 



- cutout 图像

<img src="\figures\apple_cutout.png" alt="apple_cutout" style="zoom:33%;" /> <img src=" figures\boy_cutout.png" alt="boy_cutout" style="zoom:33%;" /> <img src=" figures\elephant_cutout.png" alt="elephant_cutout" style="zoom:33%;" /> 

- cutmix 图像

<img src="figures\apple+boy_cutmix.png" alt="apple+boy_cutmix" style="zoom:33%;" /> <img src="\figures\boy+elephant_cutmix.png" alt="boy+elephant_cutmix" style="zoom:33%;" /> <img src="figures\elephant+apple_cutmix.png" alt="elephant+apple_cutmix" style="zoom:33%;" /> 

- 经归一化和标准数据增强（crop/flip）的部分图像输入

<img src="\figures\inputs.png" alt="inputs" style="zoom:80%;" /> 



- 对部分特定类别的预测准确率，图中标签82（sunflower）、30（dolphin）、36（hamster）预测非常准确，而对35（girl）准确率较低

<img src="\figures\pva.png" alt="pva" style="zoom:80%;" />  

