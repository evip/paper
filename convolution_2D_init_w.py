# dung bo loc w
# khoi tao gia tri cho convolution2D
# 1 anh co' 3 kenh mau
# dung 9 bo loc cho moi kenh mau
# out ra 27 image
from libcore import *

filters = torch.tensor( [
    [
        [0.,0.,0.,0.,0.],
        [0.,0.,0.,0.,0.],
        [0.,0.,1.,0.,0.],
        [0.,0.,0.,0.,0.],
        [0.,0.,0.,0.,0.],
    ],
    [
        [0.,0.,0.,0.,0.],
        [0.,0.,.5,0.,0.],
        [0.,.5,1.,.5,0.],
        [0.,0.,.5,0.,0.],
        [0.,0.,0.,0.,0.],
    ],
    [
        [0.,.1,.2,.1,0.],
        [.1,.3,.5,.2,.1],
        [.2,.5,1.,.5,.2],
        [.1,.3,.5,.3,.1],
        [0.,.1,.2,.1,0.],
    ],
    [
        [0.,.7,.8,.7,0.],
        [.7,1.,1.,1.,.7],
        [.8,1.,1.,1.,.8],
        [.7,1.,1.,1.,.7],
        [0.,.7,.8,.7,0.],
    ],
    [
        [0.,0.,0.,0.,0.],
        [0.,0.,0.,0.,0.],
        [0.,-1,0.,1,0.],
        [0.,0.,0.,0.,0.],
        [0.,0.,0.,0.,0.],
    ],
    [
        [0.,0.,0.,0.,0.],
        [0.,0.,0.,0.,0.],
        [0.,-1,0.,1,0.],
        [0.,0.,0.,0.,0.],
        [0.,0.,0.,0.,0.],
    ],
    [
        [0.,0.,0.,0.,0.],
        [0.,0.,1.,0.,0.],
        [0.,0.,0.,0.,0.],
        [0.,0.,-1.,0.,0.],
        [0.,0.,0.,0.,0.],
    ],
    [
        [0.,0.,0.,0.,0.],
        [0.,0.,0.,1,0.],
        [0.,0.,0.,0.,0.],
        [0.,-1,0.,0.,0.],
        [0.,0.,0.,0.,0.],
    ],
    [
        [0.,0.,0.,0.,0.],
        [0.,-1,0.,0.,0.],
        [0.,0.,0.,0.,0.],
        [0.,0.,0.,1,0.],
        [0.,0.,0.,0.,0.],
    ]
])

filters_data = filters[:, None].float()

file_name = 'blacklab.jpg'
img = Image.open(file_name)
img_tensor = Fv.to_tensor(img)[0:3]
img_data = img_tensor[:, None]

model = torch.nn.Conv2d(1, 9, kernel_size=5, padding=2)
model.weight = torch.nn.Parameter(filters_data, requires_grad=False)

bias_9 = torch.zeros(9, dtype=torch.float32)
model.bias = torch.nn.Parameter(bias_9, requires_grad=False)

result = model(img_data)
batch = img_data.size(0)
out_channel = model.out_channels

for i in range(batch):
    for j in range(out_channel):
        imshow_tensor(result[i:i+1, j:j+1, :, :])
        




