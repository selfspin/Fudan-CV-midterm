import os
import matplotlib.pyplot as plt

iter_list = []
loss_list = []
with open("training_dir/fcos_imprv_R_50_FPN_1x/log.txt") as file:
    for item in file:
        loss_location = item.find('loss:')
        if loss_location != -1:
            iter_location = item.find('iter:')
            print(item[(iter_location + 6):(loss_location - 2)], item[(loss_location + 6):(loss_location + 12)])
            iter_list.append(int(item[(iter_location + 6):(loss_location - 2)]))
            loss_list.append(float(item[(loss_location + 6):(loss_location + 12)]))

plt.plot(iter_list, loss_list)
plt.xlabel('iter')
plt.ylabel('loss')
plt.title('Train FCOS on VOC')
plt.savefig('training_dir/fcos_imprv_R_50_FPN_1x/loss.jpg')
