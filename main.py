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
loss_all, fwiou_all, val_loss_all, val_fwiou_all = solver.train(opt)
end = time.perf_counter() # 结束时间

total = end - start
hours = total//3600
minutes = total//60 - hours*60
seconds = (total - hours*3600 - minutes*60) // 1
print("总用时：{:n}小时 {:n}分钟 {:n}秒\n".format(hours, minutes, seconds), flush = True)
