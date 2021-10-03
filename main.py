# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 16:23:04 2021

@author: masteryi
"""

import time

import param
import solver



opt = param.parser()

start = time.perf_counter() # 开始时间
loss_all, fwiou_all, val_loss_all, val_fwiou_all = solver.solver(opt)

print('train loss:', loss_all, flush=True)
print('valid loss:', val_loss_all, flush=True)
print('train fwiou', fwiou_all, flush=True)
print('valid fwiou', val_loss_all, flush=True)

print('min loss in train and valid: {:.4f}, {:.4f}'.format(min(loss_all), min(val_loss_all)), flush=True)
print('max fwiou in train and valid: {:.4f}, {:.4f}'.format(max(fwiou_all), max(val_fwiou_all)), flush=True)

end = time.perf_counter() # 结束时间

total = end - start
hours = total//3600
minutes = total//60 - hours*60
seconds = (total - hours*3600 - minutes*60) // 1
print("总用时：{:n}小时 {:n}分钟 {:n}秒\n".format(hours, minutes, seconds), flush = True)