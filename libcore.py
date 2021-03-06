import torch
from torch import nn
import torch.optim as optim
from torch.nn import functional as F

import torchvision
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
from torchvision.transforms import functional as Fv

from matplotlib import pyplot as plt
from PIL import Image

import numpy as np
import itertools
from numpy.fft import fft2, ifft2
import io
import pywt 

import timeit
from time import process_time

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def imshow_img(img):
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    
def imshow_tensor(tensor):
    plt.imshow(tensor.squeeze(0).squeeze(0).numpy())
    plt.axis('off')
    plt.show()    

def imshow_tensor_interpolate(ts, scale):
    ts_interpolate = F.interpolate(ts.clone(), (32,32))
    print(ts_interpolate.shape)
    plt.imshow(ts_interpolate.squeeze(0).squeeze(0).numpy())
    plt.axis('off')
    plt.show()
    
# # https://stackoverflow.com/questions/34768717/matplotlib-unable-to-save-image-in-same-resolution-as-original-image/34769840
def imshow_actual_size(im_data, img_name):
    dpi = 80
    
    height, width =  im_data.shape
    
    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)
    
    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    
    # Hide spines, ticks, etc.
    ax.axis('off')
    
    # Display the image.
    ax.imshow(im_data, interpolation='nearest')
    
    # Add something...
    #ax.annotate('Look at This!', xy=(590, 650), xytext=(500, 500),
# =============================================================================
#     ax.annotate('Look at This!', xy=(100, 100), xytext=(70, 70),
#                 color='cyan', size=24, ha='right',
#                 arrowprops=dict(arrowstyle='fancy', fc='cyan', ec='none'))
# =============================================================================
    
    # Ensure we're displaying with square pixels and the right extent.
    # This is optional if you haven't called `plot` or anything else that might
    # change the limits/aspect.  We don't need this step in this case.
    ax.set(xlim=[0, width], ylim=[height, 0], aspect=1)
    
    #fig.savefig(img_name, dpi=dpi, transparent=True)
    plt.show()   

# n: number for stack
# d=1: dim
def upInChannel(filters, in_channel, d):
    tmp = []
    for i in range(in_channel):
        tmp.append(filters)
    return torch.stack(tmp, dim=d)

num_classes = 9
class_names = ['cr', 'gg', 'in', 'pa', 'ps', 'rp', 'rs', 'sc', 'sp']
imgsize = (64,64)
data_dir = "steel/"

data_transforms = {
    'train': transforms.Compose(
            
            [#transforms.Resize([28,28],2),
             transforms.ToTensor(),                                 
             #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
     ),
    
    'val': transforms.Compose(
            [transforms.ToTensor(),
             #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
     )
}

batch_size_train = 32
batch_size_test = 256
validation_split = .20 # 20%
num_worker = 1


# Chia tap train va test doc lap
dataset = torchvision.datasets.ImageFolder(data_dir, data_transforms['train'])
train_size = int((1-validation_split) * len(dataset))
test_size = len(dataset) - train_size
data_train, data_test = torch.utils.data.random_split(dataset, [train_size, test_size])

data_train_loader = DataLoader(data_train, batch_size=batch_size_train, shuffle=True)
data_test_loader = DataLoader(data_test, batch_size=batch_size_test)


