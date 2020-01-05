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
