# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 16:34:28 2021

@author: masteryi
"""

import deeplab.deeplabv3plus as dl
import hrocr.seg_hrnet_ocr as ho
import segmenter.factory as ft
import segformer.segformer_pytorch as sf



def build(opt):
    name = opt.model_name
    if name == 'deeplab':
        return dl.DeepLabV3Plus(n_classes=2, n_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18], multi_grids=[1, 2, 4], output_stride=16)
    elif name == 'hrocr':
        return ho.HighResolutionNet()
    elif name == 'segmenter':
        return ft.create_segmenter(patch_size=16)
    elif name == 'segformer':
        return sf.Segformer(num_classes=2)