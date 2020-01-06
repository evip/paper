from __future__ import print_function

import itertools



# Disable warnings from the Scattering transform...
import warnings
warnings.filterwarnings("ignore")

# Train and visualize the performances of our models

from model_utils import AttrDict, show, generate_image, train, test, evaluate_model, display_classified_images
import model_utils as MU

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



################################
## Thiet ke tinh tuong duong voi chuyen doi scattering
################################
from collections import OrderedDict

class ScatteringEqualNet(nn.Module):
    def __init__(self):
        "Defines the parameters of the model."
        super(ScatteringEqualNet, self).__init__()
        
# =============================================================================
#         self.scattering = nn.Sequential(OrderedDict([
#                     ('conv1', nn.Conv2d(3, 9, kernel_size=5)),
#                     ('pool1', nn.MaxPool2d(kernel_size=(2, 2))),
#                     ('relu1', nn.ReLU()),
#                     
#                     ('conv2', nn.Conv2d(9, 27, kernel_size=5)),
#                     ('pool2', nn.MaxPool2d(kernel_size=(2, 2))),
#                     ('relu2', nn.ReLU()),
#                     
#                     ('conv3', nn.Conv2d(27, 243, kernel_size=5)),
#                     ('relu3', nn.ReLU())
#                 ]))        
# 
#         #19683 = 243 * 9 * 9
#         self.fc = nn.Sequential(OrderedDict([
#             ('linear1', nn.Linear(19683, 1024)),
#             ('relu4', nn.ReLU()),
#             ('linear2', nn.Linear(1024, 512)),
#             ('relu5', nn.ReLU()),
#             ('linear3', nn.Linear(512, 256)),
#             ('relu6', nn.ReLU()),
#             ('linear4', nn.Linear(256, 9)),
#             ('relu7', nn.ReLU()),
#             #('sig7', nn.LogSoftmax(dim=-1))
#         ]))
# =============================================================================
        
        self.scattering = nn.Sequential(OrderedDict([
                    ('conv1', nn.Conv2d(3, 27, kernel_size=5, padding=2)),
                    ('pool1', nn.MaxPool2d(kernel_size=(2, 2))),
                    ('relu1', nn.ReLU()),
                    
                    ('conv2', nn.Conv2d(27, 81, kernel_size=5, padding=2)),
                    ('pool2', nn.MaxPool2d(kernel_size=(2, 2))),
                    ('relu2', nn.ReLU()),
                    
                    ('conv3', nn.Conv2d(81, 81, kernel_size=5, padding=2)),
                    ('pool3', nn.MaxPool2d(kernel_size=(2, 2))),
                    ('relu3', nn.ReLU())
                ]))     
        
# =============================================================================
#         self.convNext = nn.Sequential(OrderedDict([
#                     ('conv4', nn.Conv2d(81, 81, kernel_size=5, padding=2)),
#                     ('pool4', nn.MaxPool2d(kernel_size=(2, 2))),
#                     ('relu4', nn.ReLU())
#                 ]))
# =============================================================================
        
        #1296 = 81 * 4 * 4
        self.fc = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(1296, 1024)),
            ('relu5', nn.ReLU()),
# =============================================================================
#             ('linear2', nn.Linear(1024, 512)),
#             ('relu6', nn.ReLU()),
#             ('linear3', nn.Linear(512, 256)),
#             ('relu7', nn.ReLU()),
# =============================================================================
            ('linear4', nn.Linear(256, 9)),
            ('relu8', nn.ReLU()),
            #('sig7', nn.LogSoftmax(dim=-1))
        ]))
        
        
    def forward(self, x):

        #print('x: ', x.shape)
        
        output = self.scattering(x)
        #print('after scatter:', output.shape)
        
        output = self.convNext(output)        
        #print('after next: ', output.shape)
        
        output = output.view(-1, 1296)
        output = self.fc(output)
        return F.log_softmax(output) 
        #return output


class ScatteringEqualNet_Good(nn.Module):
    def __init__(self):
        "Defines the parameters of the model."
        super(ScatteringEqualNet_Good, self).__init__()
        
        self.scattering = nn.Sequential(OrderedDict([
                    ('conv1', nn.Conv2d(3, 27, kernel_size=5, padding=2)),
                    ('pool1', nn.MaxPool2d(kernel_size=(2, 2))),
                    ('relu1', nn.ReLU()),
                    
                    ('conv2', nn.Conv2d(27, 81, kernel_size=5, padding=2)),
                    ('pool2', nn.MaxPool2d(kernel_size=(2, 2))),
                    ('relu2', nn.ReLU()),
                    
                    ('conv3', nn.Conv2d(81, 243, kernel_size=5, padding=2)),
                    ('relu3', nn.ReLU())
                ]))             
        
        #1296 = 243 * 8 * 4
        self.classifer = nn.Sequential(OrderedDict([
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
        #print('x: ', x.shape)
        output = self.scattering(x)
        #print('after scatter:', output.shape)
        
        output = output.view(output.size(0), -1)
        
        output = self.classifer(output)
        return F.log_softmax(output) 
        #return output


class ScatteringEqualNet_Batch_Good(nn.Module):
    def __init__(self):
        "Defines the parameters of the model."
        super(ScatteringEqualNet_Batch_Good, self).__init__()
        
        self.scattering = nn.Sequential(OrderedDict([
                    ('conv1', nn.Conv2d(3, 27, kernel_size=5, padding=2)),
                    ('bn1', nn.BatchNorm2d(27)),
                    ('pool1', nn.MaxPool2d(kernel_size=(2, 2))),
                    ('relu1', nn.ReLU()),
                    
                    ('conv2', nn.Conv2d(27, 81, kernel_size=5, padding=2)),
                    ('pool2', nn.MaxPool2d(kernel_size=(2, 2))),
                    ('relu2', nn.ReLU()),
                    
                    ('conv3', nn.Conv2d(81, 243, kernel_size=5, padding=2)),
        
                    ('relu3', nn.ReLU())
                ]))             
        
        #1296 = 243 * 8 * 4
        self.classifer = nn.Sequential(OrderedDict([
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
        #print('x: ', x.shape)
        output = self.scattering(x)
        #print('after scatter:', output.shape)
        
        output = output.view(output.size(0), -1)
        
        output = self.classifer(output)
        return F.log_softmax(output) 
        #return output





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
two_fc_classifier = ScatteringEqualNet_Batch_Good()

#two_fc_classifier = Scattering2dNet()

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



