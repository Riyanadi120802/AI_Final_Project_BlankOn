from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.backends.cudnn as cudnn
import warnings
warnings.filterwarnings('ignore')
from torch import nn
from torchvision import datasets, models, transforms
from torch.optim import lr_scheduler

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy
from collections import OrderedDict

cudnn.benchmark = True
plt.ion()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Python code (.py): initialization model

class Model:
    def __init__(self, model_path:str) -> None:
        self.model_path = model_path

    def load_model(self):
        model = tf.keras.models.load_model(self.model_path) # tensorflow
        model = torch.model.load(self.model_path) # pytorch
        return model