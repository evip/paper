from __future__ import print_function
import itertools

# Disable warnings from the Scattering transform...
import warnings
warnings.filterwarnings("ignore")

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
    
two_fc_classifier = Scattering2dNet_Good()
if MU.use_cuda : two_fc_classifier.cuda()
evaluate_model(two_fc_classifier)


