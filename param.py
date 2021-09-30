# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 11:25:51 2021

@author: masteryi
"""

import argparse



def parser():
    parser = argparse.ArgumentParser()
    
    # epoch和batch
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)

    # Adma参数
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--b1", type=float, default=0.9)
    parser.add_argument("--b2", type=float, default=0.999)

    # 输入输出路径
    parser.add_argument("--input_path", type=str, default='F:\seaice_pos8')
    parser.add_argument("--image_path", type=str, default='F:\seaice_outimg')
    parser.add_argument("--model_path", type=str, default='F:\seaice_model')

    config = parser.parse_args()
    return config