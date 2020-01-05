# https://github.com/t-vi/pytorch-tvmisc/blob/master/misc/2D-Wavelet-Transform.ipynb
from libcore import *

w=pywt.Wavelet('bior2.2')
# =============================================================================
# pyplot.plot(w.dec_hi[::-1], label="dec hi")
# pyplot.plot(w.dec_lo[::-1], label="dec lo")
# pyplot.plot(w.rec_hi, label="rec hi")
# pyplot.plot(w.rec_lo, label="rec lo")
# pyplot.title("Bior 2.2 Wavelets")
# pyplot.legend()
# =============================================================================

img = Image.open("chris.jpg")
plt.imshow(img)
plt.show()

img_tensor = Fv.to_tensor(img)
#print(img_tensor.shape)

dec_hi = torch.tensor(w.dec_hi[::-1]) 
dec_lo = torch.tensor(w.dec_lo[::-1])
rec_hi = torch.tensor(w.rec_hi)
rec_lo = torch.tensor(w.rec_lo)

dec1 = dec_lo.unsqueeze(0)*dec_lo.unsqueeze(1)
dec2 = dec_lo.unsqueeze(0)*dec_hi.unsqueeze(1)
dec3 = dec_hi.unsqueeze(0)*dec_lo.unsqueeze(1)
dec4 = dec_hi.unsqueeze(0)*dec_hi.unsqueeze(1)

filter_dec = torch.stack([dec1, dec2, dec3, dec4], dim=0)

rec1 = rec_lo.unsqueeze(0)*rec_lo.unsqueeze(1)
rec2 = rec_lo.unsqueeze(0)*rec_hi.unsqueeze(1)
rec3 = rec_hi.unsqueeze(0)*rec_lo.unsqueeze(1)
rec4 = rec_hi.unsqueeze(0)*rec_hi.unsqueeze(1)

filter_rec = torch.stack([rec1, rec2, rec3, rec4], dim=0)

def conv2d(x, w):
    return torch.nn.functional.conv2d(x, w)

    
img_tensor_g = img_tensor[0:1, :, :].unsqueeze(0)

two = torch.tensor([2])    

def size_level(img, level):
    h, w = img.size(2), img.size(3)
    h_new, w_new = h/torch.pow(two, level).int(), w/torch.pow(two,level).int()
    return h_new, w_new
    
        
def wavelet_lowpass(img, level):
    for level in range(level):    
        w, h = size_level(img_tensor_g,level)
        #print('w, h: ', w, h)
        if level == 0:
            img_conv = img.clone()
        else:
            img_conv = img_conv
        #print('img_conv: ', img_conv.shape)
        #print('filter: ', filter_dec[None, 0:1, :, :].shape)
        
        img_conv = conv2d(img_conv, filter_dec[None, 3:4, :, :])        
        img_conv = F.interpolate(img_conv, (w, h))
        
        plot_actual_size(img_conv.squeeze(0).squeeze(0), '')
        #print(img_conv.shape)
    return img_conv

#[3,1,225,225]
img_data = Fv.to_tensor(img)[0:3].unsqueeze(1)

#[4,3,6,6]
filter_init = upInChannel(filter_dec, 1, 1).float()

model = torch.nn.Conv2d(1,4, kernel_size=6, padding=2)
model.weight = torch.nn.Parameter(filter_init, requires_grad=False)

result = model(img_data)
batch, out_channel = result.size(0), result.size(1)

for i in range(batch):
    for j in range(out_channel):
        imshow_tensor(result[i:i+1,j:j+1, :, :].detach().cpu())






