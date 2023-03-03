# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 17:01:22 2021

@author: masteryi
"""

from PIL import Image
import numpy as np



class Save_img3():
    def __init__(self, batch_data, outimgname):
        self.imgs, self.gt, self.predict = batch_data
        # print(self.imgs.shape, self.gt.shape, self.predict.shape)
        self.outimgname = outimgname
        
    def joint_img_vertical(self, imgs):
        # print(imgs.ndim) 之前按维度区分通道数，但是tensor类型都是4维度
        # PIL只能处理 HWC 的形状
        if imgs.shape[1] == 3:
            width, height = imgs[0][0].shape
            # print(width, height)
            result = Image.new('RGB', (width, height*imgs.shape[0]))
            # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
            # imgs = imgs.mul_(255).add_(0.5).clamp_(0, 255).numpy().transpose((0, 2, 3, 1))
            # 交换了通道也没发现什么作用...
            imgs = (imgs*255).numpy().transpose((0, 2, 3, 1))
        
        # single channel img
        elif imgs.shape[1] == 1:
            width, height = imgs[0][0].shape
            result = Image.new('L', (width, height*imgs.shape[0]))
            imgs = (imgs*255).numpy().transpose((0, 2, 3, 1)).squeeze(3)
        
        for i in range(imgs.shape[0]):
            # im = Image.fromarray(imgs[i].astype(np.uint8))
            im = Image.fromarray(np.uint8(imgs[i]))
            result.paste(im, box=(0, i * height))
         
            # 取消注释以下代码可直接保存图片
            # im = Image.fromarray((img * 255).astype(np.uint8))
            # im.save(filename)
            # result.save(filename)
        return result
    
    def joint_img_horizontal(self, img, output, mask):
        # print(img.size, output.size, mask.size)
        if img.size == output.size == mask.size:
          width, height = img.size
          result = Image.new('RGB', (width * 3, height))
          result.paste(img, box = (0, 0))
          result.paste(output, box = (width, 0))
          result.paste(mask, box = (2*width, 0))
          
        return result
    
    def save_img(self, img, filename):
        '''
        it is used to save an img in filename directory
        :param img: the img should be Image type.
        :param filename: like "**.png"
        '''
        img.save(filename)
    
    def save(self):
        batch_img = self.joint_img_vertical(self.imgs)        # img, gt, predict都是tensor类型
        batch_gt = self.joint_img_vertical(self.gt)
        batch_predict = self.joint_img_vertical(self.predict)
        img_for_save = self.joint_img_horizontal(batch_img, batch_gt, batch_predict)
        
        self.save_img(img_for_save, self.outimgname)
        