from __future__ import  absolute_import
import os
from random import sample
from torch import nn
import ipdb
import matplotlib
import numpy as np
from tqdm import tqdm
import torch
from utils.config import opt
from data.dataset import Dataset, TestDataset, inverse_normalize,TestDataset2
from model import FasterRCNNVGG16
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from utils import array_tool as at
from utils.vis_tool import visdom_bbox
from utils.eval_tool import eval_detection_voc
from torch.utils.tensorboard import SummaryWriter

import resource


#############     设备设置     ###########
# 我使用的服务器0号显卡满了，所以设置用后面的
os.environ["CUDA_VISIBLE_DEVICES"] = '2,3'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
## 占用内存设置
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# print(rlimit[1])
resource.setrlimit(resource.RLIMIT_NOFILE, (2048*2, rlimit[1]))
##########################################

matplotlib.use('agg')


def eval(dataloader, faster_rcnn, test_num=10000):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(enumerate(dataloader)):
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == test_num: break

    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)
    return result


def train(**kwargs):
    opt._parse(kwargs)
    writer= SummaryWriter(opt.tbdir) if opt.tbvis==True else 0
    dataset = Dataset(opt)
    print('load data')
    dataloader = data_.DataLoader(dataset, \
                                  batch_size=1, \
                                  shuffle=True, \
                                  pin_memory=True,
                                  num_workers=opt.num_workers)
    testset = TestDataset(opt)
    test_dataloader = data_.DataLoader(testset,
                                       batch_size=1,
                                       num_workers=opt.test_num_workers,
                                       shuffle=False, \
                                       pin_memory=True
                                       )
    testset2 = TestDataset2(opt)
    test_dataloader2 = data_.DataLoader(testset2,
                                       batch_size=1,
                                       num_workers=opt.test_num_workers,
                                       shuffle=True, \
                                       pin_memory=True
                                       )
    faster_rcnn = FasterRCNNVGG16()
    print('model construct completed')
    
    trainer = FasterRCNNTrainer(faster_rcnn).to(device)
    if opt.load_path:
        trainer.load(opt.load_path)
        print('load pretrained model from %s' % opt.load_path)
    # trainer.vis.text(dataset.db.label_names, win='labels')
    if opt.tbvis==True:
        writer.add_text('labels', ", ".join(dataset.db.label_names)) 

    best_path = None
    best_map = 0
    lr_ = opt.lr
    alli=0
    dsize=len(dataloader)
    for epoch in range(opt.epoch):
        trainer.reset_meters()
        for ii, (img, bbox_, label_, scale) in tqdm(enumerate(dataloader)):
            alli+=1
            scale = at.scalar(scale)
            img, bbox, label = img.to(device).float(), bbox_.to(device), label_.to(device)
            trainer.train_step(img, bbox, label, scale)
            
            if alli%int(opt.plot_every/20)==0:
                for iii, (img2, bbox_2, label_2, scale2) in enumerate(test_dataloader2):
                    scale2 = at.scalar(scale2)
                    img2, bbox2, label2 = img2.to(device).float(), bbox_2.to(device), label_2.to(device)
                    trainer.test_step(img2, bbox2, label2,scale2)
                    break

            if (alli) % opt.plot_every == 0 and opt.tbvis==True:
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()

                # plot loss
                # trainer.vis.plot_many(trainer.get_meter_data())
                for k, v  in trainer.get_meter_data().items():
                    writer.add_scalar('losses/'+ str(k), v, global_step=alli)

                for k, v  in trainer.get_meter_data2().items():
                    writer.add_scalar('test_losses/'+ str(k), v, global_step=alli)

                # plot groud truth bboxes
                ori_img_ = inverse_normalize(at.tonumpy(img[0]))
                gt_img = visdom_bbox(ori_img_,
                                     at.tonumpy(bbox_[0]),
                                     at.tonumpy(label_[0]))
                # trainer.vis.img('gt_img', gt_img)
                writer.add_image('img/gt_img', gt_img, global_step=alli)

                # plot proposal
                sample_roi=trainer.get_proposal(img, bbox, label, scale)
                sample_roi=at.tonumpy(sample_roi)[0:20]
                pro_img = visdom_bbox(ori_img_,sample_roi)
                writer.add_image('img/proposal_img', pro_img, global_step=alli)

                # plot predict bboxes
                _bboxes, _labels, _scores = trainer.faster_rcnn.predict([ori_img_], visualize=True)
                pred_img = visdom_bbox(ori_img_,
                                       at.tonumpy(_bboxes[0]),
                                       at.tonumpy(_labels[0]).reshape(-1),
                                       at.tonumpy(_scores[0]))
                # trainer.vis.img('pred_img', pred_img)
                writer.add_image('img/pred_img', pred_img, global_step=ii+1+epoch*dsize)
                writer.flush()
                # rpn confusion matrix(meter)
                # trainer.vis.text(str(trainer.rpn_cm.value().tolist()), win='rpn_cm')
                # roi confusion matrix
                # trainer.vis.img('roi_cm', at.totensor(trainer.roi_cm.conf, False).float())
        eval_result = eval(test_dataloader, faster_rcnn, test_num=opt.test_num)
        # trainer.vis.plot('test_map', eval_result['map'])
        if opt.tbvis==True:
            writer.add_scalar('mAP@.5', eval_result['map'],global_step=epoch+1)
        lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']
        log_info = 'lr:{}, map:{},loss:{}'.format(str(lr_),
                                                  str(eval_result['map']),
                                                  str(trainer.get_meter_data()))
        # trainer.vis.log(log_info)
        if opt.tbvis==True:
            writer.add_text('info', log_info,global_step=epoch+1)
            writer.flush()
        if eval_result['map'] > best_map:
            best_map = eval_result['map']
            best_path = trainer.save(best_map=best_map)
        if epoch%9==0 and epoch!=0:
            trainer.load(best_path)
            trainer.faster_rcnn.scale_lr(opt.lr_decay)
            lr_ = lr_ * opt.lr_decay
        # if epoch == 13: 
        #     break


            
    print('best_path:',best_path)
    print('finish training！')
    pass

if __name__ == '__main__':
    import fire

    fire.Fire()
