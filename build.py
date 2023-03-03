# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 16:34:28 2021

@author: masteryi
"""

import deeplab.deeplabv3plus as dl
import hrocr.seg_hrnet_ocr as ho
import segmenter.factory as ft
import segformer.segformer_pytorch as sf

from losses import CrossEntropyLoss2d, FocalLoss, DiceLoss, LovaszSoftmax

import torch.optim as optim



def model(opt):
    name = opt.model_name
    if name == 'deeplab':
        return dl.DeepLabV3Plus(n_classes=2, n_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18], multi_grids=[1, 2, 4], output_stride=16)
    elif name == 'hrocr':
        return ho.HighResolutionNet()
    elif name == 'segmenter':
        return ft.create_segmenter(patch_size=16)
    elif name == 'segformer':
        return sf.Segformer(num_classes=2)



def lossfunc(opt):
    neme = opt.lossfunc
    if neme == 'ce':
        return CrossEntropyLoss2d()
    if neme == 'focal':
        return FocalLoss()
    if neme == 'dice':
        return DiceLoss()
    if neme == 'lovasz':
        return LovaszSoftmax()



def optimizer(opt, model):
    name = opt.optim
    if name == 'adam':
        return optim.Adam(model.parameters(), opt.lr, (opt.b1, opt.b2))
    if name == 'sgd':
        return optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, dampening=0, weight_decay=0, nesterov=False)



def scheduler(opt, optimizer):
    name = opt.schedule
    if name == 'step':
        return optim.lr_scheduler.StepLR(optimizer, step_size=opt.step, gamma=opt.gamma, verbose=True)
    if name == 'exponential':
        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98, last_epoch=-1, verbose=True)


