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

from paper_model_utils_paper import AttrDict, show, generate_image, train, test, evaluate_model, display_classified_images
import paper_model_utils_paper as MU

MU.class_names = ['cr', 'gg', 'in', 'pa', 'ps', 'rp', 'rs', 'sc', 'sp']
#kwargs = {'num_workers': 0, 'pin_memory': False} if MU.use_cuda else {}

MU.imgsize = (64,64)
MU.args.batch_size      = 3072
MU.args.test_batch_size = 3072
MU.args.epochs = 150

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


from paper_modelNeural import ModelLevel2, C
myLayer = ModelLevel2()
x = torch.randn(MU.args.batch_size, C, MU.imgsize[0], MU.imgsize[1])
y = myLayer(x)
feature_num = y.size(1) * y.size(2) * y.size(3)


from collections import OrderedDict
from paper_classifer_paper import OurLinear_I1

# Classify
class OurClassifier_I1(nn.Module):
    def __init__(self):
        super(OurClassifier_I1, self).__init__()
        
        # extract features
        self.ourExtractFeatures = ModelLevel2()

        # Classifer one linear layer
        self.classifer = OurLinear_I1(feature_num, 9)        
    
    def forward(self, x):
        out = self.ourExtractFeatures(x)
        
        out = out.view(out.size(0), -1)
        out = self.classifer(out)
        return F.log_softmax(out)    
        

# Train & test
classifier = OurClassifier_I1()
if MU.use_cuda :
    classifier.cuda()
evaluate_model(classifier)



