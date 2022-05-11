import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

data = np.loadtxt(open("./runs/train/exp6/results.csv", "rb"), delimiter=",", skiprows=1)
train_loss = data[:, 1:4]
test_loss = data[:, 8:11]

writer = SummaryWriter('log')

for i in range(100):
    writer.add_scalars('Loss curve on training set and test set',
                      {'train': sum(train_loss[i]), 'test': sum(test_loss[i])},
                      i)


writer.flush()