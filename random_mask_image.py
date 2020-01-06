# https://twitter.com/francoisfleuret/status/1208342223340875776?s=20

import torch
import timeit
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

img = np.array(Image.open('chris.jpg'))
plt.imshow(img)
plt.show
x = torch.from_numpy(img).transpose(2,0).unsqueeze(0)

n, _, h, w = x.shape

rnd = torch.rand(2,n, 1, 2).sort(-1).values

r, c = torch.linspace(0, 1, h+2)[None, None], torch.linspace(0, 1, w+2)[None, None]

mask = (((r > rnd[0, :, :, :1]) & (r<rnd[0, :, :, 1:])).unsqueeze(-1) *
        ((c>rnd[1, :, :, :1]) & (c<rnd[1, :, :, 1:])).unsqueeze(-2))[:, :, 1:-1, 1:-1].expand_as(x)

print(mask.shape)

mask_img = mask.squeeze(0).transpose(2,0).float()


plt.imshow(mask_img.numpy())
plt.show()

