from libcore import *

# =============================================================================
# for i, (images_show, labels_show) in enumerate(data_train_loader):
#     bt, ch_in, _, _ = images_show.shape
#     print(bt, ch_in)
# =============================================================================

H, W = 64, 64

############ Fourier  ########################
filter_np = np.array([
        [0.,.7,.8,.7,0.],
        [.7,1.,1.,1.,.7],
        [.8,1.,1.,1.,.8],
        [.7,1.,1.,1.,.7],
        [0.,.7,.8,.7,0.],
    ])
h = np.zeros((H, W))
h[-2:,-2:] = filter_np[:2,:2]
h[-2:,:3]  = filter_np[:2,2:5]
h[:3,-2:]  = filter_np[2:5,:2]
h[:3,:3]   = filter_np[2:5,2:5]    
H = h/np.sum(h)
t_start_fourier = time.time()
for i, (img_j, label_j) in enumerate(data_train_loader):
    batch, channel, _, _ = img_j.shape
    for k1 in range(batch):
        for k2 in range(channel):
            np.real(ifft2(fft2(img_j[k1, k2:k2+1, :, :].squeeze(0)) * fft2(H)))
    
t_elapsed_fourier = time.time() - t_start_fourier
print('fourier time: ', t_elapsed_fourier)


############ Filter in pytorch ########################
filter_ts = torch.tensor([
        [0.,.7,.8,.7,0.],
        [.7,1.,1.,1.,.7],
        [.8,1.,1.,1.,.8],
        [.7,1.,1.,1.,.7],
        [0.,.7,.8,.7,0.],
    ])    
    
################ GPU nn.Conv2D ########################
filter_tensor_gpu = upInChannel(filter_ts[None], 3, 1)
device = "cuda:0"
model_gpu = torch.nn.Conv2d(3,1,5, padding=2, stride=1)
#print('weight: ', model_gpu.weight.shape)
model_gpu.weight = torch.nn.Parameter(filter_tensor_gpu.float(), requires_grad=False)
model_gpu.to(device)

t_start_conv2d_gpu = time.time()
for i, (images, labels) in enumerate(data_train_loader):
    data_gpu = images.to('cuda:0')
    result_conv2d_gpu = model_gpu(data_gpu)    
    #print(result_conv2d_gpu.shape)
    
t_elapsed_conv2d_gpu = time.time() - t_start_conv2d_gpu
print('GPU time: ', t_elapsed_conv2d_gpu)


################ CPU nn.Conv2D ########################
filter_tensor_cpu = upInChannel(filter_ts[None], 3, 1)
model_cpu = torch.nn.Conv2d(3,1,5, padding=2, stride=1)
#print('weight: ', model_cpu.weight.shape)
model_cpu.weight = torch.nn.Parameter(filter_tensor_cpu.float(), requires_grad=False)
t_start_conv2d_cpu = time.time()
for i, (images_cpu, labels_cpu) in enumerate(data_train_loader):    
    result_conv2d_cpu = model_cpu(images_cpu)

t_elapsed_conv2d_cpu = time.time() - t_start_conv2d_cpu
print('cpu time: ', t_elapsed_conv2d_cpu)

# =============================================================================
# # Result 
# fourier time:  4.196772575378418
# GPU time:  1.7014737129211426
# cpu time:  3.1455535888671875
# =============================================================================
































