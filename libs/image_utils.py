# -*- coding: utf-8 -*-
"""
Create on 2019/2/28 15:40
Create by ring
Function Description:
"""
from imageio import imread
import numpy as np
import torchvision.utils as tvls
import torchvision.transforms as transforms

def get_image_from_path(path):
    """
    从输入路径读取单个图像 并返回PIL Image对象
    """
    img = imread(path)
    return img

def image_to_tensor(image):
    """
    将输入的PIL Image对象转为pytorch的Tensor对象,channelximage_heightximage_width格式
    """
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return trans(image)

def get_tensor_image(path):
    img = get_image_from_path(path)
    return image_to_tensor(img)

def save_tensor_images(images,filename,nrow=None,normalize=True):
    """
    将输入的image tensor数组存为一张图片,默认nrow为8,也就是一行显示8个图片
    """
    if not nrow:
        tvls.save_image(images, filename, normalize=normalize)
    else:
        tvls.save_image(images, filename, normalize=normalize,nrow=nrow)