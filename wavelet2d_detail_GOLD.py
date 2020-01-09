from libcore import *

w=pywt.Wavelet('bior2.2')
#print(w)
dec_hi = torch.tensor(w.dec_hi[::-1]) 
dec_lo = torch.tensor(w.dec_lo[::-1])
rec_hi = torch.tensor(w.rec_hi)
rec_lo = torch.tensor(w.rec_lo)

img = Image.open('chris.jpg')
img_tensor = Fv.to_tensor(img)[0:3].unsqueeze(0)
#print(img_tensor.shape)

#imshow_actual_size(img_tensor.squeeze(0).squeeze(0).numpy(), '')

filters = torch.stack([dec_lo.unsqueeze(0)*dec_lo.unsqueeze(1),
                       dec_lo.unsqueeze(0)*dec_hi.unsqueeze(1),
                       dec_hi.unsqueeze(0)*dec_lo.unsqueeze(1),
                       dec_hi.unsqueeze(0)*dec_hi.unsqueeze(1)], dim=0)

inv_filters = torch.stack([rec_lo.unsqueeze(0)*rec_lo.unsqueeze(1),
                           rec_lo.unsqueeze(0)*rec_hi.unsqueeze(1),
                           rec_hi.unsqueeze(0)*rec_lo.unsqueeze(1),
                           rec_hi.unsqueeze(0)*rec_hi.unsqueeze(1)], dim=0)

w_L1_filter = upInChannel(filters, 1,1)
#print(w_L1_filter.size(2))


###################### Wavelet Level 1 ########################
class WaveletConvL1(torch.nn.Module):
    def __init__(self):
        super(WaveletConvL1, self).__init__()
        self.conv1 = nn.Conv2d(1,4,6, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
    
    def forward(self, x):
        output = x.view(-1, 1, x.size(2), x.size(3))
        output = self.conv1(output)
        output = self.pool1(output)
        return output
        #return output.view(-1, 3, output.size(2), output.size(3))

# =============================================================================
# # Test for 1 image
# model_img = WaveletConvL1()
# model_img.conv1.weight = torch.nn.Parameter(w_L1_filter, requires_grad=False)
# img_out = model_img(img_tensor)
# for k1 in range(3):
#     for k2 in range(4):
#         img_name = 'wavelet/level1/chanel_{}_filter_{}.png'.format(k1, k2)
#         imshow_actual_size(img_out[k1, k2].detach().cpu(), img_name)
# =============================================================================


# =============================================================================
# # Test model L1 for dataset
# model_wavelet = WaveletConvL1()
# model_wavelet.conv1.weight = torch.nn.Parameter(w_L1_filter, requires_grad=False)
# model_wavelet.to('cuda:0')
# 
# t0 = time.time()
# for i, (images, labels) in enumerate(data_train_loader):
#     data = images.to('cuda:0')
#     #data = images
#     result_wavelet = model_wavelet(data)
#     
# # =============================================================================
# #     batch, channel_out, _, _ = result_wavelet.shape        
# #     for k1 in range(batch):
# #         for k2 in range(channel_out):
# #             imshow_tensor(result_wavelet[k1:k1+1, k2:k2+1, :, :].detach().cpu())
# # =============================================================================
#             
# t = time.time() - t0
# print('Gpu time: ', t)
# =============================================================================
    


###################### Wavelet Level 2 ########################
class WaveletConvL2(torch.nn.Module):
    def __init__(self):
        super(WaveletConvL2, self).__init__()
        self.conv1 = nn.Conv2d(1,4,6, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
    
        self.conv2 = nn.Conv2d(1,4,6, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        
    def forward(self, x):
        output = x.view(-1, 1, x.size(2), x.size(3))
        
        output = self.conv1(output)
        output = self.pool1(output)
        
        output = output.view(-1, 1, output.size(2), output.size(3))
        output = self.conv2(output)
        output = self.pool2(output)
                
        return output        
    
# =============================================================================
# model_wavelet_L2 = WaveletConvL2()
# model_wavelet_L2.conv1.weight = torch.nn.Parameter(w_L1_filter, requires_grad=False)
# model_wavelet_L2.conv2.weight = torch.nn.Parameter(w_L1_filter, requires_grad=False)
# 
# # Test wavelet L2 for 1 image
# img_out = model_wavelet_L2(img_tensor)
# print(img_out.shape)
# for k1 in range(12):
#     for k2 in range(4):
#         img_name = 'wavelet/level2/wavelet/chanel_{}_filter_{}.png'.format(k1, k2)
#         imshow_actual_size(img_out[k1, k2].detach().cpu(), img_name)
# =============================================================================


###################### Wavelet Level 3 ########################
class WaveletConvL3(torch.nn.Module):
    def __init__(self):
        super(WaveletConvL3, self).__init__()
        self.conv1 = nn.Conv2d(1,4,6, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
    
        self.conv2 = nn.Conv2d(1,4,6, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        
        self.conv3 = nn.Conv2d(1,4,6, padding=2)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))
        
    def forward(self, x):
        output = x.view(-1, 1, x.size(2), x.size(3))
        
        output = self.conv1(output)
        output = self.pool1(output)
        
        output = output.view(-1, 1, output.size(2), output.size(3))
        output = self.conv2(output)
        output = self.pool2(output)
        
        output = output.view(-1, 1, output.size(2), output.size(3))
        output = self.conv3(output)
        output = self.pool3(output)
                
        return output
    
# =============================================================================
# model_wavelet_L3 = WaveletConvL3()
# model_wavelet_L3.conv1.weight = torch.nn.Parameter(w_L1_filter, requires_grad=False)
# model_wavelet_L3.conv2.weight = torch.nn.Parameter(w_L1_filter, requires_grad=False)
# model_wavelet_L3.conv3.weight = torch.nn.Parameter(w_L1_filter, requires_grad=False)
# 
# # Test wavelet L3 for 1 image
# img_out = model_wavelet_L3(img_tensor)
# print(img_out.shape)
# for k1 in range(48):
#     for k2 in range(4):
#         img_name = 'wavelet/level3/wavelet/chanel_{}_filter_{}.png'.format(k1, k2)
#         imshow_actual_size(img_out[k1, k2].detach().cpu(), img_name)
# =============================================================================
        
    
###################### Wavelet Level 4 ########################
class WaveletConvL4(torch.nn.Module):
    def __init__(self):
        super(WaveletConvL4, self).__init__()
        self.conv1 = nn.Conv2d(1,4,6, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
    
        self.conv2 = nn.Conv2d(1,4,6, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        
        self.conv3 = nn.Conv2d(1,4,6, padding=2)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))
        
        self.conv4 = nn.Conv2d(1,4,6, padding=2)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2))
        
    def forward(self, x):
        output = x.view(-1, 1, x.size(2), x.size(3))
        
        output = self.conv1(output)
        output = self.pool1(output)
        
        output = output.view(-1, 1, output.size(2), output.size(3))
        output = self.conv2(output)
        output = self.pool2(output)
        
        output = output.view(-1, 1, output.size(2), output.size(3))
        output = self.conv3(output)
        output = self.pool3(output)
                
        output = output.view(-1, 1, output.size(2), output.size(3))
        output = self.conv4(output)
        output = self.pool4(output)
        
        return output
    
model_wavelet_L4 = WaveletConvL4()
model_wavelet_L4.conv1.weight = torch.nn.Parameter(w_L1_filter, requires_grad=False)
model_wavelet_L4.conv2.weight = torch.nn.Parameter(w_L1_filter, requires_grad=False)
model_wavelet_L4.conv3.weight = torch.nn.Parameter(w_L1_filter, requires_grad=False)
model_wavelet_L4.conv4.weight = torch.nn.Parameter(w_L1_filter, requires_grad=False)

# Test wavelet L2 for 1 image
img_out = model_wavelet_L4(img_tensor)
print(img_out.shape)
for k1 in range(192):
    for k2 in range(4):
        img_name = 'wavelet/level4/wavelet/chanel_{}_filter_{}.png'.format(k1, k2)
        imshow_actual_size(img_out[k1, k2].detach().cpu(), img_name)
        
        
        
