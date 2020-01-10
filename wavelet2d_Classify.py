from libcore import *

from model_utils import AttrDict, show, generate_image, train, test, evaluate_model, display_classified_images
import model_utils as MU

MU.class_names = ['cr', 'gg', 'in', 'pa', 'ps', 'rp', 'rs', 'sc', 'sp']

MU.imgsize = (64,64)
MU.args.batch_size      = 32
MU.args.test_batch_size = 32
num_classes = 9

MU.train_loader = data_train_loader
MU.test_loader = data_test_loader


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



###################### Wavelet Level 1 --> classification ########################
##################################################################################        
class WaveletConvL1(torch.nn.Module):
    def __init__(self):
        super(WaveletConvL1, self).__init__()
        self.conv1 = nn.Conv2d(1,4,6, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
    
        self.conv1.weight = torch.nn.Parameter(w_L1_filter, requires_grad=False)
        
    def forward(self, x):
        output = x.view(-1, 1, x.size(2), x.size(3))
        output = self.conv1(output)
        output = self.pool1(output)
        return output
        #return output.view(-1, 3, output.size(2), output.size(3))

# =============================================================================
# x = torch.randn(1,3,64,64)    
# model = WaveletConvL1()
# out = model(x)
# print(out.shape)
# =============================================================================
        
class WaveletPriorClassificationL1(nn.Module):
    def __init__(self):
        super(WaveletPriorClassificationL1, self).__init__()
        
        self.classifier = nn.Sequential(OrderedDict([
            #3*4*31*31 = 11532
            ('linear1', nn.Linear(11532, 512)),
            ('relu5', nn.ReLU()),
# =============================================================================
#             ('linear2', nn.Linear(1024, 512)),
#             ('relu6', nn.ReLU()),
#             ('linear3', nn.Linear(512, 256)),
#             ('relu7', nn.ReLU()),
# =============================================================================
            ('linear4', nn.Linear(512, 9)),
            ('relu8', nn.ReLU()),
            #('sig7', nn.LogSoftmax(dim=-1))
        ]))
    
    def forward(self, x):
        x.to('cuda')
        output = decompositionModel(x)
        
        output = output.view(-1, 11532)        
        
        output = self.classifier(output)
        #return F.log_softmax(output)
        return output
    


###################### Wavelet Level 2 --> classification ########################
##################################################################################        
class WaveletConvL2(torch.nn.Module):
    def __init__(self):
        super(WaveletConvL2, self).__init__()
        self.conv1 = nn.Conv2d(1,4,6, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
    
        self.conv2 = nn.Conv2d(1,4,6, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv1.weight = torch.nn.Parameter(w_L1_filter, requires_grad=False)
        self.conv2.weight = torch.nn.Parameter(w_L1_filter, requires_grad=False)
        
    def forward(self, x):
        output = x.view(-1, 1, x.size(2), x.size(3))
        
        output = self.conv1(output)
        output = self.pool1(output)
        
        output = output.view(-1, 1, output.size(2), output.size(3))
        output = self.conv2(output)
        output = self.pool2(output)
                
        return output        
    
# =============================================================================
# x = torch.randn(1,3,64,64)    
# model = WaveletConvL2()
# out = model(x)
# print(out.shape)
# =============================================================================
    

class WaveletPriorClassificationL2(nn.Module):
    def __init__(self):
        super(WaveletPriorClassificationL2, self).__init__()      
        
        self.classifier = nn.Sequential(OrderedDict([
            #12*4*15*15 = 10800
            ('linear1', nn.Linear(10800, 512)),
            ('relu5', nn.ReLU()),
# =============================================================================
#             ('linear2', nn.Linear(1024, 512)),
#             ('relu6', nn.ReLU()),
#             ('linear3', nn.Linear(512, 256)),
#             ('relu7', nn.ReLU()),
# =============================================================================
            ('linear4', nn.Linear(512, 9)),
            ('relu8', nn.ReLU()),
            #('sig7', nn.LogSoftmax(dim=-1))
        ]))
    
    def forward(self, x):
        x.to('cuda')
        output = decompositionModel(x)
        
        output = output.view(-1, 10800)        
        
        output = self.classifier(output)
        #return F.log_softmax(output)
        return output 
        






###################### Wavelet Level 3 --> classification ########################
##################################################################################        
        
class WaveletConvL3(torch.nn.Module):
    def __init__(self):
        super(WaveletConvL3, self).__init__()
        self.conv1 = nn.Conv2d(1,4,6, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
    
        self.conv2 = nn.Conv2d(1,4,6, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        
        self.conv3 = nn.Conv2d(1,4,6, padding=2)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))
        
        self.conv1.weight = torch.nn.Parameter(w_L1_filter, requires_grad=False)
        self.conv2.weight = torch.nn.Parameter(w_L1_filter, requires_grad=False)
        self.conv3.weight = torch.nn.Parameter(w_L1_filter, requires_grad=False)

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


class WaveletPriorClassificationL3(nn.Module):
    def __init__(self):
        super(WaveletPriorClassificationL3, self).__init__()      
        
        self.classifier = nn.Sequential(OrderedDict([
            #12*4*15*15 = 9408
            ('linear1', nn.Linear(9408, 512)),
            ('relu5', nn.ReLU()),
# =============================================================================
#             ('linear2', nn.Linear(1024, 512)),
#             ('relu6', nn.ReLU()),
#             ('linear3', nn.Linear(512, 256)),
#             ('relu7', nn.ReLU()),
# =============================================================================
            # 192*4*3*3
            ('linear4', nn.Linear(512, 9)),
            ('relu8', nn.ReLU()),
            #('sig7', nn.LogSoftmax(dim=-1))
        ]))
    
    def forward(self, x):
        x.to('cuda')
        output = decompositionModel(x)        
        
        output = output.view(-1, 9408)
        
        output = self.classifier(output)
        #return F.log_softmax(output)
        return output 
        
    
        
    
###################### Wavelet Level 4 --> classification ########################
##################################################################################        
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
        
        self.conv1.weight = torch.nn.Parameter(w_L1_filter, requires_grad=False)
        self.conv2.weight = torch.nn.Parameter(w_L1_filter, requires_grad=False)
        self.conv3.weight = torch.nn.Parameter(w_L1_filter, requires_grad=False)
        self.conv4.weight = torch.nn.Parameter(w_L1_filter, requires_grad=False)
        
        
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
    

class WaveletPriorClassificationL4(nn.Module):
    def __init__(self):
        super(WaveletPriorClassificationL4, self).__init__()      
        
        self.classifier = nn.Sequential(OrderedDict([
            # 192*4*3*3 = 6912
            ('linear1', nn.Linear(6912, 512)),
            ('relu5', nn.ReLU()),
# =============================================================================
#             ('linear2', nn.Linear(1024, 512)),
#             ('relu6', nn.ReLU()),
#             ('linear3', nn.Linear(512, 256)),
#             ('relu7', nn.ReLU()),
# =============================================================================
            # 192*4*3*3
            ('linear4', nn.Linear(512, 9)),
            ('relu8', nn.ReLU()),
            #('sig7', nn.LogSoftmax(dim=-1))
        ]))
    
    def forward(self, x):
        x.to('cuda')
        output = decompositionModel(x)        
        
        output = output.view(-1, 6912)
        
        output = self.classifier(output)
        #return F.log_softmax(output)
        return output 


decompositionModel = WaveletConvL4()
model = WaveletPriorClassificationL4()
decompositionModel.to('cuda:0')
model.to('cuda:0')

print(decompositionModel.conv1.weight.data)

print(w_L1_filter)
# =============================================================================
# if MU.use_cuda : model.cuda()
# evaluate_model(model)
# =============================================================================



