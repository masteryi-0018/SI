# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 11:31:28 2021

@author: masteryi
"""

import os
import torch
# import time
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
# import torch.nn.functional as F
from torchvision.utils import save_image
from datetime import datetime
from torch.utils.data import random_split
import matplotlib.pyplot as plt

import param
import dataset
import deeplab.deeplabv3plus as dl
import hrocr.seg_hrnet_ocr as ho
import segmenter.factory as ft

from metrics import SegmentationMetric
from losses import CrossEntropyLoss2d, FocalLoss, DiceLoss, LovaszSoftmax



def solver(opt):
    # 数据集和变换
    transform = transforms.Compose([transforms.ToTensor()])
    data = dataset.Seaice(opt, transform=transform)
    
    train_size = int(0.8 * len(data))
    valid_size = len(data) - train_size
    # 3791 1624
    trainset, validset = random_split(dataset=data, lengths=[train_size, valid_size], generator=torch.Generator().manual_seed(0))
    
    # 超算 num_workers=16, pin_memory=True, drop_last=True
    trainloader = DataLoader(dataset=trainset, batch_size=opt.batch_size, shuffle=False, drop_last=True)
    validloader = DataLoader(dataset=validset, batch_size=opt.batch_size, shuffle=False, drop_last=True)
    
    
    # 模型 损失 优化器 学习率
    # model = dl.DeepLabV3Plus(n_classes=2, n_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18], multi_grids=[1, 2, 4], output_stride=16)
    # model = ho.HighResolutionNet()
    model = ft.create_segmenter(patch_size=16)
    
    # criterion = nn.CrossEntropyLoss()
    # BCEwithlogitsloss = BCELoss + Sigmoid
    # 二分类用BCE 多分类用CE
    # criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), opt.lr, (opt.b1, opt.b2))
    # Note that step should be called after validate()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True, threshold=0.001)
    
    # CE Focal Dice
    criterion = CrossEntropyLoss2d()
    # criterion = FocalLoss()
    # criterion = DiceLoss()
    # criterion = LovaszSoftmax()
    
    
    # 设定device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        model.to(device)
        criterion.to(device)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model, device_ids=[0,1,2,3])
            model.to(device)
    
    '''
    # 模型，如果epoch不为0，就加载已经训练过的ckpt
    if epoch != 0:
        ckpt = os.path.join(opt.model_path, 'model.pth')
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            model.to(device)
            criterion.to(device)
            # 多块gpu只转移模型
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model, device_ids=[0,1,2,3])
                model.to(device)
        model.load_state_dict(torch.load(ckpt))
    
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            model.to(device)
            criterion.to(device)
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model, device_ids=[0,1,2,3])
                model.to(device)
    '''
    
    
    # 开始训练
    n_batchs = len(trainloader)
    loss_all = []
    fwiou_all = []
    val_loss_all = []
    val_fwiou_all = []
    
    for epoch in range(opt.n_epochs):
        starttime = datetime.now()
        
        # 训练
        losses, fwious = train(trainloader, model, criterion, optimizer, device, opt.n_epochs, n_batchs, epoch)
        
        training_loss = sum(losses)/len(losses) # losses是列表，取所有batch的平均值作为epoch的loss
        training_fwiou = sum(fwious)/len(fwious)
        print("Train: epoch:{}/{}, loss:{:.4f}, fwiou:{:.4f}".
             format(epoch+1, opt.n_epochs, training_loss, training_fwiou), flush = True)
        
        
        
        # 验证
        val_losses, val_fwious = valid(validloader, model, criterion, optimizer, device, opt.n_epochs, epoch)
        val_loss = sum(val_losses)/len(val_losses)
        val_fwiou = sum(val_fwious)/len(val_fwious)
        print("Valid: epoch:{}/{}, loss:{:.4f}, fwiou:{:.4f}".
             format(epoch+1, opt.n_epochs, val_loss, val_fwiou), flush = True)
        # 验证完进行学习率监测
        # scheduler.step(val_loss)
        
        
        # 保存模型 每个epoch的参数
        loss_all.append(training_loss)
        fwiou_all.append(training_fwiou)
        val_loss_all.append(val_loss)
        val_fwiou_all.append(val_fwiou)
        print("Time Taken:", datetime.now()-starttime, flush = True)
        
        # os.makedirs(opt.model_path, exist_ok=True)
        pth_path = os.path.join(opt.model_path, 'model.pth')
        torch.save(model.state_dict(), pth_path)
        print('--Model Saved--\n', flush = True)
        
    return loss_all, fwiou_all, val_loss_all, val_fwiou_all



def train(trainloader, model, criterion, optimizer, device, n_epochs, n_batchs, epoch):
    running_loss = 0.0
    losses = []
    running_fwiou = 0.0
    fwious = []
    model.train()
    
    print('--Started Train and Valid--', flush = True)
    for batch_idx, data in enumerate(trainloader):
        img, gt = data[0], data[1]
        
        img = img.to(device)
        gt = gt.to(device)
            
        # forward
        optimizer.zero_grad()
        predict = model(img)
        # print(predict.dtype) torch.float32 print(predict.shape) (b 2 512 512)
            
            
        premax = torch.sigmoid(predict)
        # warnings.warn("nn.functional.sigmoid is deprecated. Use torch.sigmoid instead.")
        # 这里不能找出最大值，否则会报CUDA错误，为什么？
        # premax_value, premax = torch.max(predict, dim=1, keepdim=True)
        # torch.argmax() 只返回索引
        # print(img.shape, premax.shape, gt.shape) # shape (b 3 512 512) (b 2 512 512) (b 1 512 512)
        # print(img.dtype, premax.dtype, gt.dtype) # torch.float32 torch.float32 torch.float32
            
            
        # compute the loss
        gt = gt.squeeze(1)
        gt = gt.long()
        # gt进行 CE loss 时需要删除新加的通道，即与premax的形状不同（少了通道维）且pre应为float32 gt为long
        loss = criterion(premax, gt)
        losses.append(loss.item())
        running_loss += loss.item()
        # print(loss.item()) 是一个数值
            
            
        # backward & optimize
        loss.backward()
        optimizer.step()
        # 求完loss再转回来
        gt = gt.unsqueeze(1)
        gt = gt.float()
            
            
        # 这里要 cpu detach 再计算
        premax_value, premax = torch.max(predict, dim=1, keepdim=True)
        gt = gt.cpu().detach()
        premax = premax.cpu().detach()
        saveimg(opt, gt, premax, batch_idx, train=True)
            
        
        # 指标
        metric = SegmentationMetric(2) # 2表示有2个分类，有几个分类就填几
        metric.addBatch(premax, gt)
        fwiou = metric.Frequency_Weighted_Intersection_over_Union()
        fwious.append(fwiou)
        running_fwiou += fwiou
            
            
        # 对于batch
        print("epoch:{}/{}, batch:{}/{}, loss:{:.4f}, fwiou:{:.4f}".
              format(epoch+1, opt.n_epochs, batch_idx+1, n_batchs,
                     running_loss/(batch_idx+1), running_fwiou/(batch_idx+1)), flush = True)
    
    return losses, fwious



def valid(validloader, model, criterion, optimizer, device, n_epochs, epoch):
    val_losses = []
    val_fwious = []
    model.eval()
    
    with torch.no_grad():
        for batch_idx, data in enumerate(validloader):
            img, gt = data[0], data[1]
            
            img = img.to(device)
            gt = gt.to(device)
            
            # forward
            optimizer.zero_grad()
            predict = model(img)
            premax = torch.sigmoid(predict)
            
            gt = gt.squeeze(1)
            gt = gt.long()
            loss = criterion(premax, gt)
            val_losses.append(loss.item())
            gt = gt.unsqueeze(1)
            gt = gt.float()
            
            premax_value, premax = torch.max(predict, dim=1, keepdim=True)
            gt = gt.cpu().detach()
            premax = premax.cpu().detach()
            saveimg(opt, gt, premax, batch_idx, train=False)
            
            metric = SegmentationMetric(2)
            metric.addBatch(premax, gt)
            fwiou = metric.Frequency_Weighted_Intersection_over_Union()
            val_fwious.append(fwiou)
    
    return val_losses, val_fwious



def saveimg(opt, gt, premax, batch_idx, train=True):
    outimgpath = opt.image_path
    # img = img.cpu()
    # premax = premax * 255
    # 使用save_image不需要乘255
    if train:
        train = 'train'
    else:
        train = 'valid'
    outimg = torch.cat((gt, premax), dim=3)
    for b in range(opt.batch_size):
        outimgb = outimg[b,:,:,:]
        outimgname = os.path.join(outimgpath, train, '{}.png'.format((batch_idx*opt.batch_size+b+1)))
        save_image(outimgb , outimgname , padding=0)
        # print(img.shape, gt.shape, premax.shape, outimg.shape)




if __name__ == '__main__':
    opt = param.parser()
    
    '''
    # 官方
    # Example of target with class indices
    loss = nn.CrossEntropyLoss()
    input = torch.randn(3, 5, requires_grad=True)
    # print(input) 3*5 的tensor
    target = torch.empty(3, dtype=torch.long).random_(5)
    # print(target) 是一个 1*3 的tensor，后面的5表示数字在范围5之内随机产生
    output = loss(input, target)
    print(input.shape, target.shape, output)
    # output.backward()
    
    # Example of target with class probabilities
    input_2 = torch.randn(3, 5, requires_grad=True)
    target_2 = torch.randn(3, 5).softmax(dim=1)
    # 在 dim=1 上面softmax 相当于每一行进行softmax 即每一行之和为1
    output_2 = loss(input, target)
    print(input_2.shape, target_2.shape, output_2)
    # output.backward()
    
    # BCEloss
    m = nn.Sigmoid()
    loss = nn.BCELoss()
    input_3 = torch.randn(4, 1, 3, 3, requires_grad=True)
    target_3 = torch.empty(4, 1, 3, 3).random_(2)
    output_3 = loss(m(input_3), target_3)
    print(input_3.dtype, target_3.dtype)
    print(input_3.shape, target_3.shape, output_3)
    # output.backward()
    '''
    
    loss_all, fwiou_all, val_loss_all, val_fwiou_all = solver(opt)
    counter = [i+1 for i in range(opt.n_epochs)]
    
    plt.plot(counter, loss_all, color='blue')
    plt.plot(counter, val_loss_all, color='green')
    plt.legend(['Train Loss', 'Valid Loss'], loc='best')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    plt.savefig(os.path.join(opt.image_path, 'loss.png'))

    plt.plot(counter, fwiou_all, color='blue')
    plt.plot(counter, val_fwiou_all, color='green')
    plt.legend(['Train Fwiou', 'Valid Fwiou'], loc='best')
    plt.xlabel('Epoch')
    plt.ylabel('Fwiou')
    plt.show()
    plt.savefig(os.path.join(opt.image_path, 'fwiou.png'))
    