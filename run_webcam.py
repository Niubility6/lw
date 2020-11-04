'''
File: run_webcam.py
Project: MobilePose-PyTorch
File Created: Monday, 11th March 2019 12:47:30 am
Author: Yuliang Xiu (yuliangxiu@sjtu.edu.cn)
-----
Last Modified: Monday, 11th March 2019 12:48:49 am
Modified By: Yuliang Xiu (yuliangxiu@sjtu.edu.cn>)
-----
Copyright 2018 - 2019 Shanghai Jiao Tong University, Machine Vision and Intelligence Group
'''

import argparse
import logging
import time

import cv2
import os
import numpy as np

import torch
import torch.nn as nn
from torchvision import models

from estimator import ResEstimator
from networks import *
from network import CoordRegressionNetwork
from dataloader import crop_camera

# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

if __name__ == '__main__':


    # 解析参数  ;模型 输入图片的大小  相机

    # parser = argparse.ArgumentParser(description='MobilePose Realtime Webcam.')
    # parser.add_argument('--model', type=str, default='resnet18', choices=['mobilenetv2', 'resnet18', 'shufflenetv2', 'squeezenet'])
    # parser.add_argument('--inp_dim', type=int, default=224, help='input size')
    # parser.add_argument('--camera', type=int, default=0)
    #
    # args = parser.parse_args()
    model = 'squeezenet'
    inp_dim = 224
    camera = 0
    # load the model  夹在模型
    model_path = os.path.join("./params", model+"_%d_adam_best.t7"%inp_dim)


    net = CoordRegressionNetwork(n_locations=16, backbone=model).to("cpu")
    e = ResEstimator(model_path, net, inp_dim)

    # initial the camera
    cam = cv2.VideoCapture('3.jpg')

    # ret_val, image = cam.read()
    # image = crop_camera(image)

    while True:
        # read image from the camera and preprocess
        ret_val , image = cam.read()
        if ret_val==False:
            break
        image = crop_camera(image)


        # forward the image
        humans = e.inference(image)
        image = ResEstimator.draw_humans(image, humans, imgcopy=False)
        cv2.imshow('MobilePose Demo', image)
        cv2.waitKey(0)
        # if cv2.waitKey(1) == 27: # ESC
        #     break

    cv2.destroyAllWindows()

    # # single person rgb image test
    # image = cv2.imread("./results/test.png")
    # humans = e.inference(image)
    # image = ResEstimator.draw_humans(image, humans, imgcopy=False)
    # cv2.imwrite("./results/out.png", image)
