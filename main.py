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



opt = param.parser()

start = time.perf_counter() # 开始时间

print('Model:', opt.model_name, flush=True)
print('Loss:', opt.lossfunc, flush=True)
print('Optim:', opt.optim, flush=True)
print('Schedule:', opt.schedule, '\n', flush=True)

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