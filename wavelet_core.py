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


two_fc_classifier = ScatteringEqualNet_Batch_Good()
if MU.use_cuda : two_fc_classifier.cuda()
evaluate_model(two_fc_classifier)

