# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 09:36:53 2021

@author: masteryi
"""

from torch.utils.data import Dataset
import os
import cv2
import torchvision.transforms as transforms

import param



class Seaice(Dataset):
    def __init__(self, opt, transform=None):
        self.datapath = opt.input_path
        self.imgpath = os.path.join(self.datapath,'image')
        self.transform = transform
        
    def __getitem__(self, idx):
        imgname = os.path.join(self.datapath, 'image', '{}.tif'.format(idx+1))
        gtname = os.path.join(self.datapath, 'gt', '{}.png'.format(idx+1))
        
        '''这里可以使用常见的 PIL 或者 opencv 来进行读取'''
        # PIL 是基于 python 优化的，速度会快一些，但需要转化为 numpy
        # 一般来说 C\C++ 的实现，应该比 python 速度快一点
        
        img = cv2.imread(imgname)
        # imgtc = img.transpose(2,0,1)
        # 这里torch可以直接将cv读取的图片转化成维度合适的tensor，无需手动转换
        
        # imread 后面的参数可以是 1 0 -1 分别代表 彩色 灰度 alpha
        gt = cv2.imread(gtname, 0)
        # gttc = np.expand_dims(gt, 2)
        # 求损失时不需要将gt扩充维度
        
        # print(imgtc.shape, gttc.shape)
        
        if self.transform is not None:
            img = self.transform(img)
            gt = self.transform(gt)

        return img, gt
    
    
    def __len__(self):
        return len(os.listdir(self.imgpath))




if __name__ == '__main__':
    opt = param.parser()
    
    transform = transforms.Compose([transforms.ToTensor()])
    # 有transform时自动给gt扩充了维度
    trainset = Seaice(opt, transform=None)
    
    lens = len(trainset)
    for i in range(lens):
        print(trainset[i][0].shape, trainset[i][1].shape)