# You Only Look Once v3

## 配置环境

```bash
git clone https://github.com/selfspin/Fudan-CV
cd YOLOv3
conda create -e yolo python=3.9
conda activate yolo
pip install -r requirements.txt
```

再从百度网盘

链接：https://pan.baidu.com/s/1Sp3pnMxgVONElgTCU3GAWA 
提取码：yolo

下载 `yolov3.pt` 和 `voc_best.pt` 到 `YOLOv3` 文件夹下

## 目标检测

使用`detect.py`进行检测，结果存放在`runs/detect`

```bash
python detect.py --source path/to/image.jpg --weight voc_best.pt
```

`path/to/image.jpg`为图片存放位置，例如`data/images/bus.jpg`

## 训练

以使用VOC数据集训练为例

```bash
python train.py --data voc.yaml --weights yolov3.pt --img 640 --epochs 100 --batch-size 16
```

`yolov3.pt`为预训练参数

结果保存在`runs/train`

## 测试

测试我们在VOC数据集上训练的模型

```bash
python val.py --data voc.yaml --weights voc_best.pt --img 640
```

结果保存在`runs/val`

