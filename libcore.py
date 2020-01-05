import torch
from torch import nn
from torch.nn import functional as F

import torchvision
from torchvision.transforms import functional as Fv

from matplotlib import pyplot as plt
from PIL import Image

import numpy as np
import pywt 

def imshow_img(img):
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    
def imshow_tensor(tensor):
    plt.imshow(tensor.squeeze(0).squeeze(0).numpy())
    plt.axis('off')
    plt.show()    
