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
import random
import logging

import param
import dataset
import visualize
import buildmodel

from metrics import SegmentationMetric
from losses import CrossEntropyLoss2d, FocalLoss, DiceLoss, LovaszSoftmax, ohem_loss



def solver(opt):
    logger = createlogger(opt)
    logger.info('Model:{}, Loss:{}, Optim:{}, Schedule:{}'.format(opt.model_name, opt.lossfunc, opt.optim, opt.schedule))
    
    # 模型 损失 优化器 学习率
    # modeldct = {'deeplab':deeplab, 'hrocr':hrocr, 'segmenter':segmenter, 'segformer':segformer}
    model = buildmodel.build(opt)
    
    
    # criterion = nn.CrossEntropyLoss() criterion = nn.BCEWithLogitsLoss() criterion = nn.BCELoss()
    # 二分类用BCE 多分类用CE BCEwithlogitsloss = BCELoss + Sigmoid
    # 增加reduction="none" 使其不求平均 返回每个loss，这个在ohem中写了
    ce = CrossEntropyLoss2d()
    focal = FocalLoss()
    dice = DiceLoss()
    lovasz = LovaszSoftmax()
    lossfuncdct = {'ce':ce, 'focal':focal, 'dice':dice, 'lovasz':lovasz}
    criterion = lossfuncdct[opt.lossfunc]
    
    
    # 设定device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda:0")
    model.to(device)
    criterion.to(device) # 损失函数需要放在cuda嘛？
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        # torch.distributed.init_process_group(backend="nccl")
        model = nn.DataParallel(model) # device_ids=[0,1,2,3]
        # model = nn.parallel.DistributedDataParallel(model) # 建议使用DDP方式
    
    
    adam = optim.Adam(model.parameters(), opt.lr, (opt.b1, opt.b2))
    sgd = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, dampening=0, weight_decay=0, nesterov=False)
    optimizerdct = {'adam':adam, 'sgd':sgd}
    optimizer = optimizerdct[opt.optim]
    

    # Note that step should be called after validate()    
    step = optim.lr_scheduler.StepLR(optimizer, step_size=opt.step, gamma=opt.gamma, verbose=True)
    exponential = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98, last_epoch=-1, verbose=True)
    schedulerdct = {'step':step, 'exponential':exponential}
    scheduler = schedulerdct[opt.schedule]
    
    
    os.makedirs(opt.model_path, exist_ok=True)
    pth_path = os.path.join(opt.model_path, opt.model_name) + '.pth'
    
    
    
    # 开始训练
    print('--Started Train and Valid--\n', flush = True)
    logger.info('--Started Train and Valid--\n')
    loss_all = []
    fwiou_all = []
    val_loss_all = []
    val_fwiou_all = []
    
    for epoch in range(opt.n_epochs):
        starttime = datetime.now()
        
        trainloader, validloader = datatransform(epoch, opt)
        n_batchs = len(trainloader)
        
        # 训练
        losses, fwious = train(trainloader, model, criterion, optimizer, device, n_batchs, epoch, opt)
        training_loss = sum(losses)/len(losses) # losses是列表，取所有batch的平均值作为epoch的loss
        training_fwiou = sum(fwious)/len(fwious)
        print('Train: epoch:{}/{}, loss:{:.4f}, fwiou:{:.4f}'.
             format(epoch+1, opt.n_epochs, training_loss, training_fwiou), flush = True)
        logger.info('Train: epoch:{}/{}, loss:{:.4f}, fwiou:{:.4f}'.format(epoch+1, opt.n_epochs, training_loss, training_fwiou))
        
        
        # 验证
        val_losses, val_fwious = valid(validloader, model, criterion, optimizer, device, epoch, opt)
        val_loss = sum(val_losses)/len(val_losses)
        val_fwiou = sum(val_fwious)/len(val_fwious)
        print('Valid: epoch:{}/{}, loss:{:.4f}, fwiou:{:.4f}'.
             format(epoch+1, opt.n_epochs, val_loss, val_fwiou), flush = True)
        logger.info('Valid: epoch:{}/{}, loss:{:.4f}, fwiou:{:.4f}'.format(epoch+1, opt.n_epochs, val_loss, val_fwiou))
        # 验证完进行学习率调整
        scheduler.step()
        
        
        # 保存模型 每个epoch的参数
        loss_all.append(training_loss)
        fwiou_all.append(training_fwiou)
        val_loss_all.append(val_loss)
        val_fwiou_all.append(val_fwiou)
        print('Time Taken:', datetime.now()-starttime, '\n', flush = True)
        torch.save(model.state_dict(), pth_path)
        
    print('--Model Saved--\n', flush = True)
    logger.info('--Model Saved--\n')
    
    # logger.info('train loss =', loss_all)
    # logger.info('valid loss =', val_loss_all)
    # logger.info('train fwiou =', fwiou_all)
    # logger.info('valid fwiou =', val_loss_all)
    
    logger.info('min loss in train and valid: {:.4f}, {:.4f}'.format(min(loss_all), min(val_loss_all)))
    logger.info('max fwiou in train and valid: {:.4f}, {:.4f}'.format(max(fwiou_all), max(val_fwiou_all)))
    logging.shutdown()
    
    return loss_all, fwiou_all, val_loss_all, val_fwiou_all



def datatransform(epoch, opt):
    # 数据集和变换，随机应用一种resize
    transform_1 = transforms.Compose([transforms.ToTensor()])
    transform_2 = transforms.Compose([transforms.ToTensor(), transforms.Resize(640)])
    transform_3 = transforms.Compose([transforms.ToTensor(), transforms.Resize(384)])
    
    # if epoch % 3 == 0:
        # data = dataset.Seaice(opt, transform=transform_1)
    # elif epoch % 3 == 1:
        # data = dataset.Seaice(opt, transform=transform_2)
    # else:
        # data = dataset.Seaice(opt, transform=transform_3)
    # 随机而不是循环训练
    data = dataset.Seaice(opt, transform=random.choice([transform_1, transform_2, transform_3]))
    
    train_size = int(opt.trainrate * len(data))
    valid_size = len(data) - train_size
    # 自动分，函数很好用
    trainset, validset = random_split(dataset=data, lengths=[train_size, valid_size], generator=torch.Generator().manual_seed(1))
    
    # 超算 num_workers=16, pin_memory=True, drop_last=True
    # 这里为什么不能加num_workers=10？
    trainloader = DataLoader(dataset=trainset, batch_size=opt.batch_size, shuffle=True, drop_last=True, pin_memory=True)
    validloader = DataLoader(dataset=validset, batch_size=opt.batch_size, shuffle=True, drop_last=True, pin_memory=True)
    
    return trainloader, validloader



def train(trainloader, model, criterion, optimizer, device, n_batchs, epoch, opt):
    running_loss = 0.0
    losses = []
    running_fwiou = 0.0
    fwious = []
    model.train()
    
    for batch_idx, data in enumerate(trainloader):
        img, gt = data[0], data[1]
        # print(img.shape, gt.shape) 随机选取transform会造成img和gt的变换不一样的问题，例如图片缩放不同
        img = img.to(device)
        gt = gt.to(device)
        
        
        # forward
        optimizer.zero_grad()
        predict = model(img)
        # print(predict.dtype) # torch.float32
        # print(predict.shape) # (b 2 512 512)
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
        
        
        # 训练时使用ohem，根据epoch调整
        if epoch > opt.useohem:
            keepnum = round(opt.batch_size*0.75) if opt.batch_size != 2 else 1
            loss = ohem_loss(opt.lossfunc, premax, gt, keepnum)
        else:
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
        
        
        premax_value, premax = torch.max(predict, dim=1, keepdim=True)
        gt = gt.cpu().detach()
        premax = premax.cpu().detach()
        # 这里要 cpu detach 再计算
        if opt.saveimg and random.random() < opt.sample:
            img = img.cpu().detach()
            saveimg(opt, img, gt, premax, batch_idx, train=True)
        
        
        # 指标
        metric = SegmentationMetric(2) # 2表示有2个分类，有几个分类就填几
        metric.addBatch(premax, gt)
        fwiou = metric.Frequency_Weighted_Intersection_over_Union()
        fwious.append(fwiou)
        running_fwiou += fwiou
        
        
        # 对于batch
        if opt.printbatch:
            print("epoch:{}/{}, batch:{}/{}, loss:{:.4f}, fwiou:{:.4f}".
                  format(epoch+1, opt.n_epochs, batch_idx+1, n_batchs,
                         running_loss/(batch_idx+1), running_fwiou/(batch_idx+1)), flush = True)
    
    return losses, fwious



def valid(validloader, model, criterion, optimizer, device, epoch, opt):
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
            
            if opt.saveimg and random.random() < opt.sample:
                img = img.cpu().detach()
                saveimg(opt, img, gt, premax, batch_idx, train=False)
            
            metric = SegmentationMetric(2)
            metric.addBatch(premax, gt)
            fwiou = metric.Frequency_Weighted_Intersection_over_Union()
            val_fwious.append(fwiou)
            
    return val_losses, val_fwious



def saveimg(opt, img, gt, premax, batch_idx, train=True):
    outimgpath = opt.image_path
    # img = img.cpu()
    # premax = premax * 255
    # 使用save_image不需要乘255
    if train:
        train = 'train'
    else:
        train = 'valid'
    outimgname = os.path.join(outimgpath, train, '{}.png'.format((batch_idx+1)))
    # outimg = torch.cat((gt, premax), dim=3)
    # for b in range(opt.batch_size):
        # outimgb = outimg[b,:,:,:]
        # outimgname = os.path.join(outimgpath, train, '{}.png'.format((batch_idx*opt.batch_size+b+1)))
        # save_image(outimgb , outimgname , padding=0)
        # print(img.shape, gt.shape, premax.shape, outimg.shape)
    # 改用拼接3张的（可拼接不同通道）
    
    batch_data = (img, gt, premax)
    imgObject = visualize.Save_img3(batch_data, outimgname)
    imgObject.save()



def createlogger(opt):
    logger_name = opt.model_name + '_log.txt'
    logger = logging.getLogger(__name__)
    logger.setLevel(level = logging.INFO)
    handler = logging.FileHandler(logger_name)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger




if __name__ == '__main__':
    opt = param.parser()
    
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