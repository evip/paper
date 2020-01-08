from libcore import *

w=pywt.Wavelet('bior2.2')
#print(w)
dec_hi = torch.tensor(w.dec_hi[::-1]) 
dec_lo = torch.tensor(w.dec_lo[::-1])
rec_hi = torch.tensor(w.rec_hi)
rec_lo = torch.tensor(w.rec_lo)

img = Image.open('heart_big.png')
img_tensor = Fv.to_tensor(img)[1:2].unsqueeze(0)
#print(img_tensor.shape)

imshow_actual_size(img_tensor.squeeze(0).squeeze(0).numpy(), '')

filters = torch.stack([dec_lo.unsqueeze(0)*dec_lo.unsqueeze(1),
                       dec_lo.unsqueeze(0)*dec_hi.unsqueeze(1),
                       dec_hi.unsqueeze(0)*dec_lo.unsqueeze(1),
                       dec_hi.unsqueeze(0)*dec_hi.unsqueeze(1)], dim=0)

inv_filters = torch.stack([rec_lo.unsqueeze(0)*rec_lo.unsqueeze(1),
                           rec_lo.unsqueeze(0)*rec_hi.unsqueeze(1),
                           rec_hi.unsqueeze(0)*rec_lo.unsqueeze(1),
                           rec_hi.unsqueeze(0)*rec_hi.unsqueeze(1)], dim=0)

k = 1
w_L1_filter = upInChannel(filters, 1,1)
w_L2_filter = upInChannel(torch.cat((filters, filters, filters, filters), 0),4,1)
    
model = torch.nn.Conv2d(1, 4, 6, padding=2)
#model.weight = torch.nn.Parameter(w_filter, requires_grad=False)
# =============================================================================
# result_normal = model(img_tensor)
# for i in range(4):
#     imshow_actual_size(result_normal[0, i, :, :].detach().cpu().numpy(), ' ')
# =============================================================================

# =============================================================================
# model.weight = torch.nn.Parameter(w_filter, requires_grad=False)
# result_wavelet = model(img_tensor)
# for i in range(4):
#     imshow_actual_size(result_wavelet[0, i, :, :].detach().cpu().numpy(), ' ')
# =============================================================================


class WaveletConvL1(torch.nn.Module):
    def __init__(self):
        super(WaveletConvL1, self).__init__()
        self.conv1 = nn.Conv2d(k,4*k,6, padding=2)        
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
    
    def forward(self, x):
        output = self.conv1(x)
        output = self.pool1(output)
        
        return output

# =============================================================================
# # Test model L1
# model_wavelet = WaveletConvL1()
# model_wavelet.conv1.weight = torch.nn.Parameter(w_L1_filter, requires_grad=False)
# 
# result_wavelet = model_wavelet(img_tensor)
# print(result_wavelet.shape)
# 
# for i in range(4*k):
#     imshow_actual_size(result_wavelet[0, i, :, :].detach().cpu().numpy(), '  ')
# =============================================================================

    
class WaveletConvL2(torch.nn.Module):
    def __init__(self):
        super(WaveletConvL2, self).__init__()
        self.conv1 = nn.Conv2d(1,4,6, padding=2)        
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
    
        self.conv2 = nn.Conv2d(4,16,6, padding=2)        
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        
    def forward(self, x):
        output = self.conv1(x)
        output = self.pool1(output)
        
        output = self.conv2(output)
        output = self.pool2(output)
        
        return output
        
    
model_wavelet = WaveletConvL2()
model_wavelet.conv1.weight = torch.nn.Parameter(w_L1_filter, requires_grad=False)
#model_wavelet.conv2.weight = torch.nn.Parameter(w_L2_filter, requires_grad=False)
print(model_wavelet.conv2.weight.shape)
#print(w_L2_filter.shape)

result_wavelet = model_wavelet(img_tensor)
#print(result_wavelet.shape)

# =============================================================================
# for i in range(16):
#     imshow_actual_size(result_wavelet[0, i, :, :].detach().cpu().numpy(), '  ')
#     #print(result_wavelet_net.shape)
# 
# =============================================================================
