# https://www.math.ens.fr/~feydy/Teaching/DataScience/cnn_part_2.html
# plot: https://github.com/ludoro/dataproject/blob/master/Student_notebook.ipynb
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms, utils
from torch.autograd import Variable
import itertools

# Performance monitoring
from time import process_time
import matplotlib.pyplot as plt
import numpy as np

# Disable warnings from the Scattering transform...
import warnings
warnings.filterwarnings("ignore")

# Train and visualize the performances of our models

#from model_utils import AttrDict, show, generate_image, train, test, evaluate_model, display_classified_images
#import model_utils_wavelet as MU

from model_utils_wavelet import AttrDict, show, generate_image, train, test, evaluate_model, display_classified_images
import model_utils_wavelet as MU

#MU.display_parameters()

MU.class_names = ['cr', 'gg', 'in', 'pa', 'ps', 'rp', 'rs', 'sc', 'sp']
#kwargs = {'num_workers': 0, 'pin_memory': False} if MU.use_cuda else {}

MU.imgsize = (64,64)
MU.args.batch_size      = 128
MU.args.test_batch_size = 128

num_classes = 9
data_dir = "steel/"

data_transforms = {
    'train': transforms.Compose(
            
            [#transforms.Resize([28,28],2),
                                 transforms.ToTensor(),                                 
                                 #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                 ]),
    'val': transforms.Compose(            
            [
                    transforms.ToTensor(),
                                 #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                 ])
}

validation_split = .20

# Chia tap train va test doc lap
dataset = torchvision.datasets.ImageFolder(data_dir, data_transforms['train'])
train_size = int((1-validation_split) * len(dataset))
test_size = len(dataset) - train_size
MU.train_dataset, MU.test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Iterator for the training pass:
MU.train_loader = torch.utils.data.DataLoader( MU.train_dataset,
    batch_size=MU.args.batch_size, shuffle=True, # Data is drawn as randomized minibatches
    )                                 # Practical settings for parallel loading

# Iterator for the testing pass, with the same settings:
MU.test_loader  = torch.utils.data.DataLoader( MU.test_dataset,
    batch_size=MU.args.test_batch_size, shuffle=True, 
    )


class TwoFullNet(nn.Module) :
    """
    Implements a simplistic perceptron with 3 layers :
    - one input, of size 28x28 (MNIST dataset)
    - one hidden, of size N
    - one output, of size 10 (number of classes)
    There is no built-in regularization, and we model the two
    transformations input->hidden and hidden->output as
    Linear->ReLu and Linear->SoftMax operators,
    i.e. as Fully connected computational graphs.
    The trainable parameters are the weights of the matrices
    (+ biases) involved in the "Linear" (Affine, really) operators.
    """
    def __init__(self, N) :
        "Defines the parameters of the model."
        super(TwoFullNet, self).__init__()
        
        # Linear (i.e. fully connected) layer, a matrix of size (28*28)xN
        self.fc1        = nn.Linear(3*MU.imgsize[0]*MU.imgsize[1], N)
        # Linear (i.e. fully connected) layer, a matrix of size Nx10 (10 classes as output)
        self.fc2        = nn.Linear( N, 9)

    def forward(self, x) :
        """
        Apply the model to some input data x.
        You can think of x as an image of size 28x28, but it is
        actually an Mx28x28 tensor, where M is the size of the
        mini-batch.
        """
        x = x.view(-1, 3 * MU.imgsize[0]*MU.imgsize[1]) # Turns our image into a vector
        x = self.fc1( x )     # Linear transformation
        x = F.relu(   x )     # Non-linearity (Relu = "positive part", a typical choice)
        x = self.fc2( x )     # Second linear transformation
        # Really, the softmax is the classification label, but for numerical stability,
        # all computations are made in the log domain
        return F.log_softmax(x) 


class TwoConvTwoFullNet(nn.Module) :
    """
    Implements a trainable model which is the concatenation
    of two convolutional layers + two fully connected layers.
    The choice of architecture here was mostly done at random,
    for illustrative purpose...
    """
    def __init__(self) :
        super(TwoConvTwoFullNet, self).__init__()
        # First conv operator : 30 1x5x5-filters + 30 bias constants 
        # which map an image of size WxH to a 3D volume of size 30xWxH
        # (modulo a padding argument)
        self.conv1      = nn.Conv2d( 3, 30, kernel_size=5)
        # Second conv operator : 30 10x5x5-filters + 30 bias constants
        # which map a 3D volume of size 30xWxH to a 3D volume of size 30xWxH
        # (modulo a padding argument)
        self.conv2      = nn.Conv2d(30, 30, kernel_size=5, groups=6)
        # Dropout layer : probabilistic regularization trick
        self.conv2_drop = nn.Dropout2d()
        # Linear (i.e. fully connected) layer, a matrix of size (30*11*11)x100
        self.fc1        = nn.Linear(30*13*13, 100)
        # Linear (i.e. fully connected) layer, a matrix of size 100x10 (10 classes as output)
        self.fc2        = nn.Linear( 100, 9)

    def forward(self, x) :
        "Stacks up the network layers, with the simplistic relu nonlinearity in-between."
        x = F.max_pool2d(F.relu(                self.conv1(x)),  2)
        x = F.max_pool2d(F.relu(self.conv2_drop(self.conv2(x))), 2)
        # Up to this point, the treatment of x has been roughly translation-invariant:
        # Conv2d operators and ReLu nonlinearities are completely T-I,
        # whereas the subsampling "max_pool2d" operators are 
        # As we believe that the large-scale information should not be completely
        # discarded (some features such as heels just happen to always be located in the bottom 
        # right corners of our images...), we end our pipeline (transform) with 
        # a regular two-layers perceptrons that processes the reduced image x
        # as a features vector.
        
        # At this point, x is a 3D volume of size 30xWxH.
        # Due to convolution truncatures and subsamplings, 
        # we have W=H=11, so the following operation...
        x = x.view(-1, 30*13*13)    # Turns it into a vector
        x = F.relu(   self.fc1(x))  # 1x100 vector
        x = F.dropout(x, training=self.training) # Add a dropout pass during training only
        x = self.fc2( x)            # 1x10 vector
        # Really, the softmax is the classification label, but for numerical stability,
        # all computations are made in the log domain
        return F.log_softmax(x) 


from collections import OrderedDict

#####################################################
# Scattering
#####################################################        

device = torch.device("cuda:0")

class Scattering2dNet(nn.Module):
    def __init__(self):
        super(Scattering2dNet, self).__init__()
        #self.bn = nn.BatchNorm2d(512)        
# =============================================================================
#         self.conv4 = nn.Conv2d(81, 81, kernel_size=5, padding=2)
#         self.bn1 = nn.BatchNorm2d(81)   
#         self.pool4 = nn.MaxPool2d(kernel_size=2)
#         self.relu4 = nn.ReLU()
# =============================================================================

        self.convNext = nn.Sequential(OrderedDict([
                    ('conv4', nn.Conv2d(243, 243, kernel_size=5, padding=2)),
                    #('bn1', nn.BatchNorm2d(243)),
                    ('pool4', nn.MaxPool2d(kernel_size=(2, 2))),
                    ('relu4', nn.ReLU())
                ]))
        
        #1296 = 243 * 4 * 4
        self.fc = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(15552, 1024)),
            ('relu5', nn.ReLU()),
            ('linear2', nn.Linear(1024, 512)),
            ('relu6', nn.ReLU()),
            ('linear3', nn.Linear(512, 256)),
            ('relu7', nn.ReLU()),
            ('linear4', nn.Linear(256, 9)),
            ('relu8', nn.ReLU()),
            #('sig7', nn.LogSoftmax(dim=-1))
        ]))
    
    def forward(self, x):
        #print(x.shape)    
        #output = self.bn(x.view(-1, 512, 8, 8).to(device))
        output = x.view(-1, 243, 16, 16).to(device)
        #print(output.shape)
        
        output = self.convNext(output)
    
        output = output.view(output.size(0), -1)
        
        output = self.fc(output)
        #return output
        return F.log_softmax(output) 


# Giam so luong layer
class Scattering2dNet_Good(nn.Module):
    def __init__(self):
        super(Scattering2dNet_Good, self).__init__()        
        #self.bn = nn.BatchNorm2d(512)        
# =============================================================================
#         self.conv4 = nn.Conv2d(81, 81, kernel_size=5, padding=2)
#         self.bn1 = nn.BatchNorm2d(81)   
#         self.pool4 = nn.MaxPool2d(kernel_size=2)
#         self.relu4 = nn.ReLU()
# =============================================================================

# =============================================================================
#         self.convNext = nn.Sequential(OrderedDict([
#                     ('conv4', nn.Conv2d(243, 243, kernel_size=5, padding=2)),
#                     #('bn1', nn.BatchNorm2d(243)),
#                     ('pool4', nn.MaxPool2d(kernel_size=(2, 2))),
#                     ('relu4', nn.ReLU())
#                 ]))
# =============================================================================
        
        #1296 = 243 * 4 * 4
        self.fc = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(62208, 512)),
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
        #print(x.shape)    
        #output = self.bn(x.view(-1, 512, 8, 8).to(device))
        output = x.view(-1, 243, 16, 16).to(device)
        #print(output.shape) [128,243,16,16]
        
        #output = self.convNext(output)
    
        output = output.view(output.size(0), -1)
        
        output = self.fc(output)
        #return output
        return F.log_softmax(output) 
    

# Giam so luong layer
class Scattering2dNet_Good_Cuda(nn.Module):
    def __init__(self):
        super(Scattering2dNet_Good_Cuda, self).__init__()        
        #self.bn = nn.BatchNorm2d(512)        
# =============================================================================
#         self.conv4 = nn.Conv2d(81, 81, kernel_size=5, padding=2)
#         self.bn1 = nn.BatchNorm2d(81)   
#         self.pool4 = nn.MaxPool2d(kernel_size=2)
#         self.relu4 = nn.ReLU()
# =============================================================================

# =============================================================================
#         self.convNext = nn.Sequential(OrderedDict([
#                     ('conv4', nn.Conv2d(243, 243, kernel_size=5, padding=2)),
#                     #('bn1', nn.BatchNorm2d(243)),
#                     ('pool4', nn.MaxPool2d(kernel_size=(2, 2))),
#                     ('relu4', nn.ReLU())
#                 ]))
# =============================================================================
        
        #1296 = 243 * 4 * 4
        self.classifer = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(62208, 1024)),
            ('relu5', nn.ReLU()),
# =============================================================================
#             ('linear2', nn.Linear(10240, 1024)),
#             ('relu6', nn.ReLU()),
# =============================================================================
            ('linear3', nn.Linear(1024, 512)),
            ('relu7', nn.ReLU()),
            ('linear4', nn.Linear(512, 256)),
            ('relu8', nn.ReLU()),
            ('linear5', nn.Linear(256, 9)),
            ('relu9', nn.ReLU()),
            #('sig7', nn.LogSoftmax(dim=-1))
        ]))
    
    def forward(self, x):
        #print(x.shape)    
        #output = self.bn(x.view(-1, 512, 8, 8).to(device))
        output = x.view(-1, 243, 16, 16).to(device)
        #print(output.shape) [128,243,16,16]
        
        #output = self.convNext(output)
    
        output = output.view(output.size(0), -1)
        
        output = self.classifer(output)
        #return output
        return F.log_softmax(output) 



# Giam so luong layer
class Scattering2dNet_Good_Cuda_hiden(nn.Module):
    def __init__(self):
        super(Scattering2dNet_Good_Cuda_hiden, self).__init__()        
        #self.bn = nn.BatchNorm2d(512)        
# =============================================================================
#         self.conv4 = nn.Conv2d(81, 81, kernel_size=5, padding=2)
#         self.bn1 = nn.BatchNorm2d(81)   
#         self.pool4 = nn.MaxPool2d(kernel_size=2)
#         self.relu4 = nn.ReLU()
# =============================================================================

# =============================================================================
#         self.convNext = nn.Sequential(OrderedDict([
#                     ('conv4', nn.Conv2d(243, 243, kernel_size=5, padding=2)),
#                     #('bn1', nn.BatchNorm2d(243)),
#                     ('pool4', nn.MaxPool2d(kernel_size=(2, 2))),
#                     ('relu4', nn.ReLU())
#                 ]))
# =============================================================================
        self.enhance = nn.Sequential(OrderedDict([
            ('conv3', nn.Conv2d(243, 243, kernel_size=3, padding=1)),
            #('pool3', nn.MaxPool2d(kernel_size=(2, 2))),
            #('dropout1', nn.Dropout2d()),
            ('relu3', nn.ReLU()),
            
            ('conv4', nn.Conv2d(243, 243, kernel_size=3, padding=1)),
            #('pool4', nn.MaxPool2d(kernel_size=(2, 2))),
            #('dropout2', nn.Dropout2d()),
            ('relu4', nn.ReLU()),            
        ]))      
    
        #1296 = 243 * 4 * 4
        self.classifer = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(62208, 1024)),
            #('linear1', nn.Linear(31104, 1024)),
            #('linear1', nn.Linear(15552, 1024)),
            ('relu5', nn.ReLU()),
# =============================================================================
#             ('linear2', nn.Linear(10240, 1024)),
#             ('relu6', nn.ReLU()),
# =============================================================================
            ('linear3', nn.Linear(1024, 512)),
            ('relu7', nn.ReLU()),
            ('linear4', nn.Linear(512, 256)),
            ('relu8', nn.ReLU()),
            ('linear5', nn.Linear(256, 9)),
            ('relu9', nn.ReLU()),
            #('sig7', nn.LogSoftmax(dim=-1))
        ]))
    
        torch.nn.init.xavier_uniform_(self.enhance[0].weight)
        torch.nn.init.xavier_uniform_(self.enhance[2].weight)
        
    
    def forward(self, x):
        #print(x.shape)    
        #output = self.bn(x.view(-1, 512, 8, 8).to(device))
        output = x.view(-1, 243, 16, 16).to(device)
        #print(output.shape) [128,243,16,16]
        
        output = self.enhance(output)
    
        output = output.view(output.size(0), -1)
        
        output = self.classifer(output)
        #return output
        return F.log_softmax(output) 
    
    
# =============================================================================
# from kymatio import Scattering2D
# import math        
# 
# scattering = scattering_train = Scattering2D(J=2, shape=(64, 64))
# model = Scattering2dNet()
# 
# x = torch.rand(1,3,64,64)
# x_scatter = scattering(x)
# outp = model(x_scatter)
# print(outp)
# =============================================================================

#two_fc_classifier = TwoFullNet(100)
#two_fc_classifier = TwoConvTwoFullNet()    
#two_fc_classifier = ScatteringEqualNet()

two_fc_classifier = Scattering2dNet_Good_Cuda_hiden()
if MU.use_cuda : two_fc_classifier.cuda()
evaluate_model(two_fc_classifier)

# =============================================================================
# #from scatwave.scattering import Scattering 
# from kymatio import Scattering2D
# import kymatio.datasets as scattering_datasets
# 
# #scat = Scattering2D(M=MU.imgsize[0]+8, N=MU.imgsize[1]+8, J=4, jit=True)
# scat = Scattering2D(J=4, shape= (MU.imgsize[0]+8, MU.imgsize[1]+8), L=8)
# if MU.use_cuda : scat = scat.cuda()
# 
# print(scat['psi'])
# 
# class scatteringfullnet(nn.module) :
#     """
#     implements a trainable model which is the concatenation
#     of a scattering transform + two fully connected layers.
#     """
#     def __init__(self) :
#         super(scatteringfullnet, self).__init__()
#         self.pad        = nn.zeropad2d(4)
#         self.conv1      = nn.conv2d(417, 64, kernel_size=3, padding=1)
#         #self.conv2      = nn.conv2d(30,  30, kernel_size=5, padding=5)
#         #self.conv3      = nn.conv2d(30,  30, kernel_size=5, padding=5)
#         self.fc1        = nn.linear(64*2*2, 100)
#         self.fc2        = nn.linear( 100, 10)
# 
#     def forward(self, x) :
#         "stacks up the network layers, with the simplistic relu nonlinearity in-between."
#         x = self.pad(x)
#         x = variable(scat(x.data).squeeze(1))
#         #print(x.size())
#         x = f.relu(f.max_pool2d( self.conv1(x),  2 ))
#         #x = f.relu(f.max_pool2d( self.conv2(x),  2 ))
#         #x = f.relu(f.max_pool2d( self.conv3(x),  2 ))
#         #print(x.size())
#         x = x.view(-1, 64*2*2)
#         x = f.relu(   self.fc1(x) )
#         x = f.dropout(x, training=self.training)
#         x = self.fc2( x)
#         return f.log_softmax(x) 
# 
# scattering_classifier = scatteringfullnet()
# if mu.use_cuda : scattering_classifier.cuda()
# 
# evaluate_model(scattering_classifier)
# =============================================================================



