from libcore import *

name = 'heart_big.png'

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
# show image
imshow_tensor(data)

######### nn.Conv2D ##########################
# [1,1,5,5]    
filter_tensor = filter_ts[None, None]
# [1,1,158, 158]
result_conv2d = conv2d(data, filter_tensor)
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



