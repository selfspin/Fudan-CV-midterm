from torchvision.datasets import CIFAR100
from torchtoolbox.transform import Cutout
import matplotlib.pyplot as plt

from utils import *

# load data
cifar_data_path = 'data/cifar100'
cifar_train = CIFAR100(cifar_data_path, train=True)

img_apple, apple = cifar_train[2]
img_person, person = cifar_train[3]
img_elephant, elephant = cifar_train[9]
plt.imshow(img_apple)
plt.axis('off')
plt.savefig('apple')
plt.show()
plt.imshow(img_person)
plt.axis('off')
plt.savefig('person')
plt.show()
plt.imshow(img_elephant)
plt.axis('off')
plt.savefig('elephant')
plt.show()

# mixup
lam = 0.5
img_a = np.copy(img_apple)
img_p = np.copy(img_person)
img_e = np.copy(img_elephant)
img_a_p = lam * img_a + (1 - lam) * img_p
img_a_e = lam * img_a + (1 - lam) * img_e
img_p_e = lam * img_p + (1 - lam) * img_e
plt.imshow(img_a_p.astype('uint8'))
plt.axis('off')
plt.savefig('apple+person_mixup')
plt.show()
plt.imshow(img_a_e.astype('uint8'))
plt.axis('off')
plt.savefig('apple+elephant_mixup')
plt.show()
plt.imshow(img_p_e.astype('uint8'))
plt.axis('off')
plt.savefig('person+elephant_mixup')
plt.show()

# cutout
cutout = Cutout(1)
img_apple_cutout = cutout(img_apple)
img_person_cutout = cutout(img_person)
img_elephant_cutout = cutout(img_elephant)
plt.imshow(img_apple_cutout)
plt.axis('off')
plt.savefig('apple_cutout')
plt.show()
plt.imshow(img_person_cutout)
plt.axis('off')
plt.savefig('person_cutout')
plt.show()
plt.imshow(img_elephant_cutout)
plt.axis('off')
plt.savefig('elephant_cutout')
plt.show()

# cutmix
bbx1, bby1, bbx2, bby2 = rand_bbox((1, 3, 32, 32), lam)
img_ap_cutmix = np.copy(img_a)
img_ap_cutmix[bbx1:bbx2, bby1:bby2, :] = img_p[bbx1:bbx2, bby1:bby2, :]
bbx1, bby1, bbx2, bby2 = rand_bbox((1, 3, 32, 32), lam)
img_pe_cutmix = np.copy(img_p)
img_pe_cutmix[bbx1:bbx2, bby1:bby2, :] = img_e[bbx1:bbx2, bby1:bby2, :]
bbx1, bby1, bbx2, bby2 = rand_bbox((1, 3, 32, 32), lam)
img_ea_cutmix = np.copy(img_e)
img_ea_cutmix[bbx1:bbx2, bby1:bby2, :] = img_a[bbx1:bbx2, bby1:bby2, :]
plt.imshow(img_ap_cutmix)
plt.axis('off')
plt.savefig('apple+person_cutmix')
plt.show()
plt.imshow(img_pe_cutmix)
plt.axis('off')
plt.savefig('person+elephant_cutmix')
plt.show()
plt.imshow(img_ea_cutmix)
plt.axis('off')
plt.savefig('elephant+apple_cutmix')
plt.show()
