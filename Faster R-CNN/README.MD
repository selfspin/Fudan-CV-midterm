# A Simple and Fast Implementation of Faster R-CNN

## 0. mAP

VGG16 train on `trainval` and test on `test` split. 

|                  Implementation                  |  mAP  |
| :----------------------------------------------: | :---: |
| [origin paper](https://arxiv.org/abs/1506.01497) | 0.699 |
|                       ours                       | 0.705 |

## 1. Install dependencies

```sh
# create conda env
conda create --name sfrcnn python=3.7
conda activate simp

# install pytorch 
# Note: change verison if cuda is different!
conda install pytorch torchvision cudatoolkit=11.3

# install other dependancy
pip install visdom scikit-image tqdm fire ipdb matplotlib torchnet tb-nightly
pip install tensorboardX

```

## 2. Prepare data

#### Pascal VOC2007

1. Download the training, validation, test data and VOCdevkit

   ```Bash
   wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
   wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
   wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
   ```

2. Extract all of these tars into one directory named `VOCdevkit`

   ```Bash
   tar xvf VOCtrainval_06-Nov-2007.tar
   tar xvf VOCtest_06-Nov-2007.tar
   tar xvf VOCdevkit_08-Jun-2007.tar
   ```

3. It should have this basic structure

   ```Bash
   $VOCdevkit/                           # development kit
   $VOCdevkit/VOCcode/                   # VOC utility code
   $VOCdevkit/VOC2007                    # image sets, annotations, etc.
   # ... and several other directories ...
   ```

4. modify `voc_data_dir` attribute in `utils/config.py`, or pass it to program using argument like `--voc-data-dir=/path/to/VOCdevkit/VOC2007/` .

## 3. Training


```bash
python train.py train 
```

you may refer to `utils/config.py` for more argument.

Some Key arguments:

- `--caffe-pretrain=False`: use pretrain model from caffe or torchvision (Default: torchvison)
- `--plot-every=n`: visualize prediction, loss etc every `n` batches.
- `--tbdir`: tensorboard env for visualization
- `--voc_data_dir`: where the VOC data stored
- `--use-drop`: use dropout in RoI head, default False
- `--use-Adam`: use Adam instead of SGD, default SGD. (You need set a very low `lr` for Adam)
- `--load-path`: pretrained model path, default `None`, if it's specified, it would be loaded.

To visualize some training details, you may run tensorboard by using:

```python
tensorboard --logdir TB_log
```

**NOTE**: check the path of TB_log file dir if the visual interface has nothing!



## 4. Checking

To check the the effect of the model, run following:

```python
python checking.py
```

**NOTE**: modifiy load_path attribute in checking.py to the model you want to check!

To visualize, run following:

```python
tensorboard --logdir TB_log
```

## Acknowledgement
This work builds on many excellent works, which include:

- [A simplified implemention of Faster R-CNN that replicate performance from origin paper](https://github.com/chenyuntc/simple-faster-rcnn-pytorch)(mainly)
- [Yusuke Niitani's ChainerCV](https://github.com/chainer/chainercv) 
- [Ruotian Luo's pytorch-faster-rcnn](https://github.com/ruotianluo/pytorch-faster-rcnn) which based on [Xinlei Chen's tf-faster-rcnn](https://github.com/endernewton/tf-faster-rcnn)
- [faster-rcnn.pytorch by Jianwei Yang and Jiasen Lu](https://github.com/jwyang/faster-rcnn.pytorch).It mainly refer to [longcw's faster_rcnn_pytorch](https://github.com/longcw/faster_rcnn_pytorch)
- All the above Repositories have referred to [py-faster-rcnn by Ross Girshick and Sean Bell](https://github.com/rbgirshick/py-faster-rcnn)  either directly or indirectly. 

