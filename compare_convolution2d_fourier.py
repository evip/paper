import torch
from torch.nn import functional as F
from torchvision.transforms import functional as Fv
import numpy as np
from numpy.fft import fft2, ifft2
import sys 

np.set_printoptions(threshold=sys.maxsize)

from matplotlib import pyplot as plt
from PIL import Image
name = 'heart_big.png'

def imshow_img(img):
    plt.imshow(img)
    plt.axis('off')
    plt.show()

#[b,c, m, n]
def imshow_tensor(ts):
    plt.imshow(ts.squeeze(0).squeeze(0))
    plt.axis('off')
    plt.show()

def conv2d(data, filters):
    return F.conv2d(data, filters, padding=2)


img = Image.open(name)

H, W = img.size
img_tensor = Fv.to_tensor(img)[0:1, :, :]
print('ts: ', img_tensor.shape)
data = img_tensor[None]
# [1, 158, 158]

filter_np = np.array([
        [0.,.7,.8,.7,0.],
        [.7,1.,1.,1.,.7],
        [.8,1.,1.,1.,.8],
        [.7,1.,1.,1.,.7],
        [0.,.7,.8,.7,0.],
    ])
    
filter_ts = torch.tensor([
        [0.,.7,.8,.7,0.],
        [.7,1.,1.,1.,.7],
        [.8,1.,1.,1.,.8],
        [.7,1.,1.,1.,.7],
        [0.,.7,.8,.7,0.],
        
# =============================================================================
#         [0.,0.,0.,0.,0.],
#         [0.,-1,0.,0.,0.],
#         [0.,0.,0.,0.,0.],
#         [0.,0.,0.,1,0.],
#         [0.,0.,0.,0.,0.],
# =============================================================================
        
    ])
    
# [1,1,5,5]    
filter_tensor = filter_ts[None, None]
# [1,1,158, 158]
result_conv2d = conv2d(data, filter_tensor)
imshow_tensor(data)
imshow_tensor(result_conv2d)

############ Fourier  ########################
h = np.zeros((H, W))
h[-2:,-2:] = filter_np[:2,:2]
h[-2:,:3]  = filter_np[:2,2:5]
h[:3,-2:]  = filter_np[2:5,:2]
h[:3,:3]   = filter_np[2:5,2:5]    

H = h/np.sum(h)
result_fourier = np.real(ifft2(fft2(img_tensor.squeeze(0)) * fft2(H)))
imshow_img(result_fourier)

result = result_fourier - result_conv2d.squeeze(0).squeeze(0).numpy()


