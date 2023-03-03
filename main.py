# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 16:23:04 2021

@author: masteryi
"""

import time
import matplotlib.pyplot as plt
import os

import param
import solver
from solver import train, valid, createlogger


import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from datetime import datetime
from torch.utils.data import random_split
import random
import logging
import torch.multiprocessing as mp

import dataset
import build



def plot(loss_all, fwiou_all, val_loss_all, val_fwiou_all):
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



'''use DDP to train'''
def ddp_solver(rank, world_size, opt):
    logger = createlogger(opt)
    logger.info('Model:{}, Loss:{}, Optim:{}, Schedule:{}'.format(opt.model_name, opt.lossfunc, opt.optim, opt.schedule))

    model = build.model(opt)
    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '5678'
    
    print(f"[{os.getpid()}] Initializing {rank}/{world_size} at DIST_DEFAULT_INIT_METHOD")
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    print(f"[{os.getpid()}] Computing {rank}/{world_size} at DIST_DEFAULT_INIT_METHOD")
    
    # 设定device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    torch.cuda.set_device(rank)
    model.cuda(rank)
    # 和下面应该相同
    # device = torch.device("cuda", rank)
    # model = model.to(device)
    
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    
    criterion = build.lossfunc(opt)
    optimizer = build.optimizer(opt, model)
    scheduler = build.scheduler(opt, optimizer)
    
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
            
        if opt.multi == True:
            trainloader, validloader = ddp_datatransform_multi(opt)
        else:
            trainloader, validloader = ddp_datatransform(opt)
            
        trainloader.sampler.set_epoch(epoch)
        validloader.sampler.set_epoch(epoch)
            
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
        
        print(f"[{os.getpid()}] Epoch-{epoch} ended {rank}/{world_size} at DIST_DEFAULT_INIT_METHOD on {model.device}")
            
        # 保存模型 每个epoch的参数
        loss_all.append(training_loss)
        fwiou_all.append(training_fwiou)
        val_loss_all.append(val_loss)
        val_fwiou_all.append(val_fwiou)
        print('Time Taken:', datetime.now()-starttime, '\n', flush = True)
        torch.save(model.state_dict(), pth_path)
    
    print(f"[{os.getpid()}] Finishing {rank}/{world_size} at DIST_DEFAULT_INIT_METHOD on {model.device}")
    
    print('--Model Saved--\n', flush = True)
    logger.info('--Model Saved--\n')
    
    logger.info('min loss in train and valid: {:.4f}, {:.4f}'.format(min(loss_all), min(val_loss_all)))
    logger.info('max fwiou in train and valid: {:.4f}, {:.4f}'.format(max(fwiou_all), max(val_fwiou_all)))
    logging.shutdown()
    
    return loss_all, fwiou_all, val_loss_all, val_fwiou_all



def ddp_datatransform(opt):
    # 数据集和变换，随机应用一种resize
    transform = transforms.Compose([transforms.ToTensor()])

    data = dataset.Seaice(opt, transform=transform)
    batch_size = opt.batch_size // torch.cuda.device_count()
    
    train_size = int(opt.trainrate * len(data))
    valid_size = len(data) - train_size
    
    trainset, validset = random_split(dataset=data, lengths=[train_size, valid_size], generator=torch.Generator().manual_seed(0))
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    valid_sampler = torch.utils.data.distributed.DistributedSampler(validset) # 对于val_loader, 一般不需要使用上述sampler, 只要保留原始的dataloader代码即可。
    
    trainloader = DataLoader(dataset=trainset, batch_size=batch_size, sampler=train_sampler, drop_last=True, pin_memory=True)
    validloader = DataLoader(dataset=validset, batch_size=batch_size, sampler=valid_sampler, drop_last=True, pin_memory=True)
    
    return trainloader, validloader



def ddp_datatransform_multi(opt):
    # 数据集和变换，随机应用一种resize
    transform_1 = transforms.Compose([transforms.ToTensor()])
    transform_2 = transforms.Compose([transforms.ToTensor(), transforms.Resize(640)])
    transform_3 = transforms.Compose([transforms.ToTensor(), transforms.Resize(384)])

    data = dataset.Seaice(opt, transform=random.choice([transform_1, transform_2, transform_3]))
    batch_size = opt.batch_size // torch.cuda.device_count()
    
    train_size = int(opt.trainrate * len(data))
    valid_size = len(data) - train_size
    # 自动分，函数很好用
    trainset, validset = random_split(dataset=data, lengths=[train_size, valid_size], generator=torch.Generator().manual_seed(0))
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    valid_sampler = torch.utils.data.distributed.DistributedSampler(validset) # 对于val_loader, 一般不需要使用上述sampler, 只要保留原始的dataloader代码即可。
    
    # 超算 num_workers=16, pin_memory=True, drop_last=True
    # 这里为什么不能加num_workers=10？
    trainloader = DataLoader(dataset=trainset, batch_size=batch_size, sampler=train_sampler, drop_last=True, pin_memory=True)
    validloader = DataLoader(dataset=validset, batch_size=batch_size, sampler=valid_sampler, drop_last=True, pin_memory=True)
    
    return trainloader, validloader



def ddp_main(opt):
    world_size = torch.cuda.device_count()
    # spawn默认会为函数传入一个i，且i在[0, nprocs)之间。即worker函数收到的参数列表是(i, args, )。
    # i = 0,1,2,3
    # PyTorch引入了torch.multiprocessing.spawn，可以使得单卡、DDP下的外部调用一致，即不用使用torch.distributed.launch。
    # python main.py一句话搞定DDP模式。
    tic = time.time()
    mp.spawn(ddp_solver,
        args=(world_size, opt),
        nprocs=world_size,
        join=True)
    toc = time.time()
    print(f"Finished in {toc-tic:.2f}s")




if __name__ == '__main__':
    opt = param.parser()
    
    start = time.perf_counter() # 开始时间
    
    print('Model:', opt.model_name, flush=True)
    print('Loss:', opt.lossfunc, flush=True)
    print('Optim:', opt.optim, flush=True)
    print('Schedule:', opt.schedule, '\n', flush=True)
    
    if opt.ddp == True:
        loss_all, fwiou_all, val_loss_all, val_fwiou_all = ddp_main(opt)
    else:
        loss_all, fwiou_all, val_loss_all, val_fwiou_all = solver.solver(opt)
    
    
    print('train loss =', loss_all, flush=True)
    print('valid loss =', val_loss_all, flush=True)
    print('train fwiou =', fwiou_all, flush=True)
    print('valid fwiou =', val_loss_all, flush=True)
    
    print('min loss in train and valid: {:.4f}, {:.4f}'.format(min(loss_all), min(val_loss_all)), flush=True)
    print('max fwiou in train and valid: {:.4f}, {:.4f}'.format(max(fwiou_all), max(val_fwiou_all)), flush=True)
    
    end = time.perf_counter() # 结束时间
    
    total = end - start
    hours = total//3600
    minutes = total//60 - hours*60
    seconds = (total - hours*3600 - minutes*60) // 1
    print('总用时：{:n}小时 {:n}分钟 {:n}秒\n'.format(hours, minutes, seconds), flush = True)
    
    if opt.plot:
        plot(loss_all, fwiou_all, val_loss_all, val_fwiou_all)
        