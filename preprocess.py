# -*- coding: utf-8 -*-
"""
Preprocess images: scale to 200x400x1

Created on Sun May  6 11:04:01 2018

@author: Cathey Wang
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import math
import pickle

from skimage.transform import resize


def global_variables():
    global h, w
    h = 200
    w = 400


def get_images(path):
    files = os.listdir(path)
    imgs = np.zeros((0,h,w))
    #num = 1
    for file in files:
        #print(num)
        filename = os.path.join(path, file)
        im = mpimg.imread(filename)
        if len(im.shape) == 3:
            im = np.mean(im, axis = -1)
        im = resize(im, (h, w))         # rescale
        im = np.resize(im, (1, h, w))   # add dimension
        imgs = np.concatenate((imgs, im), axis = 0)
        #num = num + 1
    return imgs


if __name__ == "__main__":
    global_variables()
    
    train_path = "../data/train/"
    train_imgs = get_images(train_path)
    
    f = open('train_imgs.pckl', 'wb')
    pickle.dump(train_imgs, f)
    f.close()