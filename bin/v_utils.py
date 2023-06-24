# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models

import numpy as np
import matplotlib.pyplot as plt 
import io
import sys
import os
from datetime import datetime

import v_def
from v_common import DimType

class ImageManager():
    """
    ImageManager
    """
    def __init__(self, dtype = DimType.dim2d):
        """
        __init__
        """
        self.images = []
        self.labels = []
        self.predicts = []

    def _transform_array(self):
        """
        _transform_array
        """
        x = np.asarray(self.images)
        y1 = np.asarray(self.labels)
        y2 = np.asarray(self.predicts)
        return x, y1, y2

    def add_one_image(self, image, label, predict):
        """
        add_one_image
        """
        self.images.append(image)
        self.labels.append(label)
        self.predicts.append(predict)

    def show_multi_images(self):
        """
        show_multi_images
        """
        images, labels, predicts = self._transform_array()
        size = len(images)

        for i in range(size):
            title = 't:{},p:{}' % (labels[i].item(), predicts[i].item())
            plt.title(title)
            plt.imshow(images[i], cmap='gray')
            plt.show()





