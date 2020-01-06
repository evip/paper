# luu image va matrix cua filter low-pass
from libcore import *

from kymatio.scattering2d.filter_bank import filter_bank
from kymatio.scattering2d.utils import fft2

M = 64
J = 4
L = 8
filters_set = filter_bank(M, M, J, L=L)

low_pass = filters_set['phi']

## Low-pass ###
print(list(low_pass))
for i in range(J):
    print(low_pass[i].shape)
    f_r = low_pass[i][..., 0].numpy()
    f_i = low_pass[i][..., 1].numpy()
    f = f_r + 1j*f_i
    filter_c = fft2(f)
    filter_c = np.fft.fftshift(filter_c)
    filter_c = np.abs(filter_c)
    #print('filter: ', filter_c.shape)

    filter_value = 'filter_scatter/lowpass/matrix_filter_lowpass_Level_{}.pt'.format(i)
    torch.save(filter_c, filter_value)    
    
    name_file = 'filter_scatter/lowpass/filter_lowpass_Level_{}'.format(i)    
    imshow_actual_size(filter_c, name_file)
    
