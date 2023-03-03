# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 11:25:51 2021

@author: masteryi
"""

import argparse



def parser():
    parser = argparse.ArgumentParser()
    
    # epoch和batch
    parser.add_argument("--n_epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=32)
    
    # 模型
    parser.add_argument("--model_name", type=str, default='deeplab')
    
    # loss
    parser.add_argument("--lossfunc", type=str, default='lovasz')
    parser.add_argument("--useohem", type=int, default=150)
    
    # 优化器
    parser.add_argument("--optim", type=str, default='adam')
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--b1", type=float, default=0.9)
    parser.add_argument("--b2", type=float, default=0.999)
    parser.add_argument("--momentum", type=float, default=0.9)
    
    # 学习率
    parser.add_argument("--schedule", type=str, default='exponential')
    parser.add_argument("--step", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=0.5)
    
    # 超算
    parser.add_argument("--printbatch", type=bool, default=False)
    parser.add_argument("--plot", type=bool, default=False)
    parser.add_argument("--saveimg", type=bool, default=False)
    
    # 数据集
    parser.add_argument("--trainrate", type=float, default=0.8)
    parser.add_argument("--sample", type=float, default=0.01)
    
    # 训练策略
    parser.add_argument("--multi", type=bool, default=False)
    parser.add_argument("--ddp", type=bool, default=True)
    
    # 路径
    parser.add_argument("--input_path", type=str, default='/project/gaoyi/seaice_all16_plus')
    parser.add_argument("--model_path", type=str, default='/project/gaoyi/seaice_model')
    parser.add_argument("--image_path", type=str, default='/project/gaoyi/seaice_model')
    
    config = parser.parse_args()
    return config
