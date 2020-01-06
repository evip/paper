# luu image va matrix cua filter low-pass
from libcore import *

from kymatio.scattering2d.filter_bank import filter_bank
from kymatio.scattering2d.utils import fft2

M = 64
J = 4
L = 8
filters_set = filter_bank(M, M, J, L=L)

band_pass = filters_set['psi']

#print(band_pass[5][0].shape)

# =============================================================================
# unit=0
# for inc in range(32):
#     
#     if inc < 8:
#         unit = 0        
#     elif inc < 16:
#         unit = 1        
#     elif inc < 24:
#         unit = 2    
#     else:
#         unit = 3
#         
#     #print(band_pass[inc][unit].shape)
#     
#     
#     f_r = band_pass[inc][unit][...,0].numpy()
#     f_i = band_pass[inc][unit][..., 1].numpy()
#     f = f_r + 1j*f_i
#     filter_c = fft2(f)
#     filter_c = np.fft.fftshift(filter_c)
#     filter_c = np.abs(filter_c)
#     
#     #print(filter_c.shape)
#     
#     theta = band_pass[inc]['theta']
#     #print(theta)
#     
#     filter_value = 'filter_scatter/bandpass/matrix_filter_bandpass_Level_{}_Theta_{}.pt'.format(unit+1, theta)
#     torch.save(filter_c, filter_value) 
#     
#     name_file = 'filter_scatter/bandpass/filter_bandpass_Level_{}_Theta_{}'.format(unit+1, theta)  
#     imshow_actual_size(filter_c, name_file)
# =============================================================================

i=0
for filter in filters_set['psi']:
    f_r = filter[0][...,0].numpy()
    f_i = filter[0][..., 1].numpy()
    f = f_r + 1j*f_i
    filter_c = fft2(f)
    filter_c = np.fft.fftshift(filter_c)
    filter_c = np.abs(filter_c)
    
    #print(filter_c.shape)
    
    theta = filter['theta']
    #print(filter['theta'])
    
    level = i // 8

# =============================================================================
#     filter_value = 'filter_scatter/bandpass/matrix_filter_bandpass_Level_{}_Theta_{}.pt'.format(level+1, theta)
#     torch.save(filter_c, filter_value) 
#     
#     name_file = 'filter_scatter/bandpass/filter_bandpass_Level_{}_Theta_{}'.format(level+1, theta)  
#     imshow_actual_size(filter_c, name_file)
# =============================================================================
    
    
    # Resize filter     
    if i < 8:
        result = filter_c[28:-28, 28:-28]
    elif i < 16:
        result = filter_c[24:-24, 24:-24]
    elif i < 24:
        result = filter_c[16:-16, 16:-16]
    else:
        result = filter_c
    
    #print(result.shape)
    
    filter_value = 'filter_scatter/bandpass/matrix_filter_bandpass_Level_{}_Theta_{}.pt'.format(4-level, theta)
    torch.save(result, filter_value) 
    
    name_file = 'filter_scatter/bandpass/filter_bandpass_Level_{}_Theta_{}'.format(4-level, theta)  
    imshow_actual_size(result, name_file)
    
    i = i+1
    
# =============================================================================
# # Structure PSI - Band Pass
# ['j', 'theta', 0]
# ['j', 'theta', 0]
# ['j', 'theta', 0]
# ['j', 'theta', 0]
# ['j', 'theta', 0]
# ['j', 'theta', 0]
# ['j', 'theta', 0]
# ['j', 'theta', 0]
# ['j', 'theta', 0, 1]
# ['j', 'theta', 0, 1]
# ['j', 'theta', 0, 1]
# ['j', 'theta', 0, 1]
# ['j', 'theta', 0, 1]
# ['j', 'theta', 0, 1]
# ['j', 'theta', 0, 1]
# ['j', 'theta', 0, 1]
# ['j', 'theta', 0, 1, 2]
# ['j', 'theta', 0, 1, 2]
# ['j', 'theta', 0, 1, 2]
# ['j', 'theta', 0, 1, 2]
# ['j', 'theta', 0, 1, 2]
# ['j', 'theta', 0, 1, 2]
# ['j', 'theta', 0, 1, 2]
# ['j', 'theta', 0, 1, 2]
# ['j', 'theta', 0, 1, 2, 3]
# ['j', 'theta', 0, 1, 2, 3]
# ['j', 'theta', 0, 1, 2, 3]
# ['j', 'theta', 0, 1, 2, 3]
# ['j', 'theta', 0, 1, 2, 3]
# ['j', 'theta', 0, 1, 2, 3]
# ['j', 'theta', 0, 1, 2, 3]
# ['j', 'theta', 0, 1, 2, 3]
# 
# =============================================================================
    
    
    