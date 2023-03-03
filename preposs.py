# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 09:10:55 2021

@author: masteryi
"""

import cv2
import os
import numpy as np
import random

import dataset
import param



def sizeto512(opt):
    trainset = dataset.Seaice(opt, transform=None)
    outpath = 'F:\seaice_512'
    c1 = c2 = c3 = 0
    
    size = 512
    name = 1
    for i in range(len(trainset)):
        if trainset[i][0].shape[1] == 512:
            # print('512')
            img = trainset[i][0]
            gt = trainset[i][1]
            
            cv2.imwrite(os.path.join(outpath, 'image', '{}.tif'.format(name)), img)
            cv2.imwrite(os.path.join(outpath, 'gt', '{}.png'.format(name)), gt)
            name += 1
            c1 += 1
        
        elif trainset[i][0].shape[1] == 1024:
            # imgbig = cv2.imread(os.path.join(datapath, 'image', '{}.tif'.format(i+1)))
            # gtbig = cv2.imread(os.path.join(datapath, 'gt', '{}.png'.format(i+1)))
            # 已经是读取完成的，不需要重复读取
            
            imgbig = trainset[i][0]
            gtbig = trainset[i][1]
            
            # 裁剪坐标为[y0:y1, x0:x1]
            for r in range(2):
                for c in range(2):
                    # print('1024')
                    img = imgbig[size*r:size*(r+1), size*c:size*(c+1)]
                    gt = gtbig[size*r:size*(r+1), size*c:size*(c+1)]
                    
                    cv2.imwrite(os.path.join(outpath, 'image', '{}.tif'.format(name)), img)
                    cv2.imwrite(os.path.join(outpath, 'gt', '{}.png'.format(name)), gt)
                    name += 1
            c2 += 1
            
        elif trainset[i][0].shape[1] == 2048:
            # imgbig = cv2.imread(os.path.join(datapath, 'image', '{}.tif'.format(i+1)))
            # gtbig = cv2.imread(os.path.join(datapath, 'gt', '{}.png'.format(i+1)))
            
            imgbig = trainset[i][0]
            gtbig = trainset[i][1]
            
            # 裁剪坐标为[y0:y1, x0:x1]
            for r in range(4):
                for c in range(4):
                    # print('2048')
                    img = imgbig[size*r:size*(r+1), size*c:size*(c+1)]
                    gt = gtbig[size*r:size*(r+1), size*c:size*(c+1)]
                    
                    cv2.imwrite(os.path.join(outpath, 'image', '{}.tif'.format(name)), img)
                    cv2.imwrite(os.path.join(outpath, 'gt', '{}.png'.format(name)), gt)
                    name += 1
            c3 += 1
                    
    
    # print(c1, c2, c3)
    # 1412 66 22
    # 固定尺寸后：66*4 + 22*16 = 264 + 352 = 616
    # 删减后 1359 66 22 共 1412+616-53=1975



def augment(opt):
    trainset = dataset.Seaice(opt, transform=None)
    outpath = 'F:\seaice_all8'
    
    negtive = 0
    name = 1
    for i in range(len(trainset)):
        img = trainset[i][0]
        gt = trainset[i][1]
        
        # 判断np数组是否全0（全0表示为负样本）
        if np.all(gt == 0):
            # negtive += 1
            cv2.imwrite(os.path.join(outpath, 'image', '{}.tif'.format(name)), img)
            cv2.imwrite(os.path.join(outpath, 'gt', '{}.png'.format(name)), gt)
            name += 1
            negtive += 1
        
        else:
            # print(img.type) 'numpy.ndarray' object has no attribute 'type'
            # np.rot90(m, k=1, axes=(0, 1))
            # 参数：m：输入的矩阵或者图像
            # k：逆时针旋转多少个 90 度，k 取 0、1、2、3 分别对应逆时针旋转 0 度、90 度、180 度、270 度
            # axes：选择两个维度进行旋转
            cv2.imwrite(os.path.join(outpath, 'image', '{}.tif'.format(name)), img)
            cv2.imwrite(os.path.join(outpath, 'gt', '{}.png'.format(name)), gt)
            name += 1
            
            img90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            gt90 = cv2.rotate(gt, cv2.ROTATE_90_CLOCKWISE)
            cv2.imwrite(os.path.join(outpath, 'image', '{}.tif'.format(name)), img90)
            cv2.imwrite(os.path.join(outpath, 'gt', '{}.png'.format(name)), gt90)
            name += 1
            
            img180 = cv2.rotate(img, cv2.ROTATE_180)
            gt180 = cv2.rotate(gt, cv2.ROTATE_180)
            cv2.imwrite(os.path.join(outpath, 'image', '{}.tif'.format(name)), img180)
            cv2.imwrite(os.path.join(outpath, 'gt', '{}.png'.format(name)), gt180)
            name += 1
            
            img270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            gt270 = cv2.rotate(gt, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imwrite(os.path.join(outpath, 'image', '{}.tif'.format(name)), img270)
            cv2.imwrite(os.path.join(outpath, 'gt', '{}.png'.format(name)), gt270)
            name += 1
            
            # 水平翻转后经过3次旋转
            img_h = cv2.flip(img, 1)
            gt_h = cv2.flip(gt, 1)
            cv2.imwrite(os.path.join(outpath, 'image', '{}.tif'.format(name)), img_h)
            cv2.imwrite(os.path.join(outpath, 'gt', '{}.png'.format(name)), gt_h)
            name += 1
            
            img_h90 = cv2.rotate(img_h, cv2.ROTATE_90_CLOCKWISE)
            gt_h90 = cv2.rotate(gt_h, cv2.ROTATE_90_CLOCKWISE)
            cv2.imwrite(os.path.join(outpath, 'image', '{}.tif'.format(name)), img_h90)
            cv2.imwrite(os.path.join(outpath, 'gt', '{}.png'.format(name)), gt_h90)
            name += 1
            
            img_h180 = cv2.rotate(img_h, cv2.ROTATE_180)
            gt_h180 = cv2.rotate(gt_h, cv2.ROTATE_180)
            cv2.imwrite(os.path.join(outpath, 'image', '{}.tif'.format(name)), img_h180)
            cv2.imwrite(os.path.join(outpath, 'gt', '{}.png'.format(name)), gt_h180)
            name += 1
            
            img_h270 = cv2.rotate(img_h, cv2.ROTATE_90_COUNTERCLOCKWISE)
            gt_h270 = cv2.rotate(gt_h, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imwrite(os.path.join(outpath, 'image', '{}.tif'.format(name)), img_h270)
            cv2.imwrite(os.path.join(outpath, 'gt', '{}.png'.format(name)), gt_h270)
            name += 1
        
        # print(name)
    print(negtive, len(trainset)-negtive)
    # 1544 484
    # pos8：只有正样本增强8倍 484*8=3872  3872/1544=2.5
    # all8：正样本增强8倍 负样本不增强 484*8=3872  3872/1544=2.5
    # 删减后 1500 475 475*8=3800



def augment_pos(opt):
    trainset = dataset.Seaice(opt, transform=None)
    outpath = 'F:\seaice_pos'
    
    # negtive = 0
    name = 1
    for i in range(len(trainset)):
        img = trainset[i][0]
        gt = trainset[i][1]
        
        # 判断np数组是否全0（全0表示为负样本）
        if not np.all(gt == 0):
            # negtive += 1
            cv2.imwrite(os.path.join(outpath, 'image', '{}.tif'.format(name)), img)
            cv2.imwrite(os.path.join(outpath, 'gt', '{}.png'.format(name)), gt)
            name += 1
        
        print(name)
    # pos 只有正样本



def augment_bsc(opt):
    trainset = dataset.Seaice(opt, transform=None)
    outpath = 'F:\seaice_all16'
    
    # negtive = 0
    name = 1
    for i in range(len(trainset)):
        img = trainset[i][0]
        gt = trainset[i][1]
        
        # 判断np数组是否全0（全0表示为负样本）
        if np.all(gt == 0):
            # negtive += 1
            cv2.imwrite(os.path.join(outpath, 'image', '{}.tif'.format(name)), img)
            cv2.imwrite(os.path.join(outpath, 'gt', '{}.png'.format(name)), gt)
            name += 1
        
        else:
            cv2.imwrite(os.path.join(outpath, 'image', '{}.tif'.format(name)), img)
            cv2.imwrite(os.path.join(outpath, 'gt', '{}.png'.format(name)), gt)
            name += 1
            
            img_bsc = colorjitter(img)
            # gt不变
            cv2.imwrite(os.path.join(outpath, 'image', '{}.tif'.format(name)), img_bsc)
            cv2.imwrite(os.path.join(outpath, 'gt', '{}.png'.format(name)), gt)
            name += 1
        
        print(name)
    # all8：正样本增强16倍 负样本不增强 484*16=7744  7744/1544=5
    # 删减后 475*16=7600 负1500 共9100



# 亮度 对比度 饱和度
def colorjitter(img):
    '''
    ### Different Color Jitter ###
    img: image
    cj_type: {b: brightness, s: saturation, c: constast}
    '''
    # 每次随机一项处理
    cj_dict = {1:'b', 2:'s', 3:'c'}
    idx = random.randint(1, 3)
    cj_type = cj_dict[idx]
    if cj_type == "b":
        # value = random.randint(-50, 50)
        value = np.random.choice(np.array([-50, -40, -30, 30, 40, 50]))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        if value >= 0:
            lim = 255 - value
            v[v > lim] = 255
            v[v <= lim] += value
        else:
            lim = np.absolute(value)
            v[v < lim] = 0
            v[v >= lim] -= np.absolute(value)

        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img
    
    elif cj_type == "s":
        # value = random.randint(-50, 50)
        value = np.random.choice(np.array([-50, -40, -30, 30, 40, 50]))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        if value >= 0:
            lim = 255 - value
            s[s > lim] = 255
            s[s <= lim] += value
        else:
            lim = np.absolute(value)
            s[s < lim] = 0
            s[s >= lim] -= np.absolute(value)

        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img
    
    elif cj_type == "c":
        brightness = 10
        contrast = random.randint(40, 100)
        dummy = np.int16(img)
        dummy = dummy * (contrast/127+1) - contrast + brightness
        dummy = np.clip(dummy, 0, 255)
        img = np.uint8(dummy)
        return img



# 高斯噪声 椒盐噪声
def noisy(img, noise_type="gauss"):
    '''
    ### Adding Noise ###
    img: image
    cj_type: {gauss: gaussian, sp: salt & pepper}
    '''
    if noise_type == "gauss":
        image=img.copy() 
        mean=0
        st=0.7
        gauss = np.random.normal(mean,st,image.shape)
        gauss = gauss.astype('uint8')
        image = cv2.add(image,gauss)
        return image
    
    elif noise_type == "sp":
        image=img.copy() 
        prob = 0.05
        if len(image.shape) == 2:
            black = 0
            white = 255            
        else:
            colorspace = image.shape[2]
            if colorspace == 3:  # RGB
                black = np.array([0, 0, 0], dtype='uint8')
                white = np.array([255, 255, 255], dtype='uint8')
            else:  # RGBA
                black = np.array([0, 0, 0, 255], dtype='uint8')
                white = np.array([255, 255, 255, 255], dtype='uint8')
        probs = np.random.random(image.shape[:2])
        image[probs < (prob / 2)] = black
        image[probs > 1 - (prob / 2)] = white
        return image




if __name__ == '__main__':
    opt = param.parser()
    # sizeto512(opt)
    # augment(opt)
    # augment_bsc(opt)
    augment_pos(opt)