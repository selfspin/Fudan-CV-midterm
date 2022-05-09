from __future__ import  absolute_import
import os
from random import sample
from torch import nn
import ipdb
import matplotlib
from tqdm import tqdm
import torch
from utils.config import opt
from data.dataset import Dataset, TestDataset, inverse_normalize ,TestDataset2,SelfDataset
from model import FasterRCNNVGG16
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from utils import array_tool as at
from utils.vis_tool import visdom_bbox,fig4vis
from utils.eval_tool import eval_detection_voc
from torch.utils.tensorboard import SummaryWriter
from data.util import  read_image
import sys
import resource

device=torch.device('cuda:0')
writer= SummaryWriter(sys.path[0]+'/'+opt.tbdir)


def vis_on_testdata(**kwargs):
    opt._parse(kwargs)
    dataset = Dataset(opt)
    print('load data')
    dataloader = data_.DataLoader(dataset, \
                                  batch_size=1, \
                                  shuffle=True, \
                                  pin_memory=True,
                                  num_workers=opt.num_workers)
    testset = TestDataset2(opt)
    test_dataloader = data_.DataLoader(testset,
                                       batch_size=1,
                                       num_workers=opt.test_num_workers,
                                       shuffle=False, \
                                       pin_memory=True
                                       )
    faster_rcnn = FasterRCNNVGG16()
    print('model construct completed')
    
    trainer = FasterRCNNTrainer(faster_rcnn).to(device)
    if opt.load_path:
        trainer.load(opt.load_path)
        print('load pretrained model from %s' % opt.load_path)
    ## 测试集上画proposal
    for ii, (imgs, gt_bboxes_, gt_labels_,scale) in tqdm(enumerate(test_dataloader)):
        img, bbox, label = imgs.to(device).float(), gt_bboxes_.to(device), gt_labels_.to(device)
        if  opt.tbvis!=True or (ii + 1) >10:
            break

        # plot groud truth bboxes
        ori_img_ = inverse_normalize(at.tonumpy(img[0]))
        gt_img = visdom_bbox(ori_img_,
                                at.tonumpy(bbox[0]),
                                at.tonumpy(label[0]))
        # trainer.vis.img('gt_img', gt_img)
        writer.add_image('test_img/gt_img', gt_img, global_step=ii+1)

        # plot proposal
        sample_roi=trainer.get_proposal(img, bbox, label, 1)
        # sample_roi=at.tonumpy(sample_roi)[0:20]
        sample_roi=at.tonumpy(sample_roi)
        pro_img = visdom_bbox(ori_img_,sample_roi)
        writer.add_image('test_img/proposal_img', pro_img, global_step=ii+1)

        # plot proposal 20
        sample_roi=trainer.get_proposal(img, bbox, label, 1)
        sample_roi=at.tonumpy(sample_roi)[0:20]
        # sample_roi=at.tonumpy(sample_roi)
        pro_img = visdom_bbox(ori_img_,sample_roi)
        writer.add_image('test_img/proposal_img_20', pro_img, global_step=ii+1)

        # plot predict bboxes
        _bboxes, _labels, _scores = trainer.faster_rcnn.predict([ori_img_], visualize=True)
        pred_img = visdom_bbox(ori_img_,
                                at.tonumpy(_bboxes[0]),
                                at.tonumpy(_labels[0]).reshape(-1),
                                at.tonumpy(_scores[0]))
        # trainer.vis.img('pred_img', pred_img)
        writer.add_image('test_img/pred_img', pred_img, global_step=ii+1)

    print('vis on test data: Done!')
    pass

def real_pred(img_dir,**kwargs):
    opt._parse(kwargs)
    
    faster_rcnn = FasterRCNNVGG16()
    print('model construct completed')
    trainer = FasterRCNNTrainer(faster_rcnn).to(device)
    if opt.load_path:
        trainer.load(opt.load_path)
        print('load pretrained model from %s' % opt.load_path)

    
    
    for ii, (img) in enumerate(os.listdir(img_dir)):
        img = read_image(img_dir + '/' + img)
        img0 = img
        img = torch.from_numpy(img)[None]
        # plot origin image
        ori_img_ = visdom_bbox(at.tonumpy(img[0]),[],)
        writer.add_image('self_img/ori_img', ori_img_, global_step=ii+1)

        # plot predict bboxes
        _bboxes, _labels, _scores = trainer.faster_rcnn.predict(img, visualize=True)
        pred_img = visdom_bbox(at.tonumpy(img[0]),
                                at.tonumpy(_bboxes[0]),
                                at.tonumpy(_labels[0]).reshape(-1),
                                at.tonumpy(_scores[0]))
        writer.add_image('self_img/pred_img', pred_img, global_step=ii+1)

    writer.flush()

    print('predict real pic: Done!')

if __name__=='__main__':
    load_path=sys.path[0]+'/checkpoints/fasterrcnn_05071517_0.7058461237091669'
    voc_data_dir =sys.path[0]+'/VOCdevkit/VOC2007/'
    vis_on_testdata(load_path=load_path,voc_data_dir=voc_data_dir)


    path = sys.path[0]+'/images'
    real_pred(img_dir=path,load_path=load_path)
