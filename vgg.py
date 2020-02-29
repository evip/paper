from libcore import *
from torchvision.models import vgg19_bn, vgg16, vgg11


from model_utils_cpu import AttrDict, show, generate_image, train, test, evaluate_model, display_classified_images
import model_utils_cpu as MU

MU.class_names = ['cr', 'gg', 'in', 'pa', 'ps', 'rp', 'rs', 'sc', 'sp']

MU.imgsize = (64,64)
MU.args.batch_size      = 16
MU.args.test_batch_size = 64
num_classes = 9

MU.train_loader = data_train_loader
MU.test_loader = data_test_loader

model = vgg11()
#model = vgg16()
#model = vgg19_bn()
model.classifier[6] = nn.Linear(in_features=4096, out_features=num_classes, bias=True)

#print(model)

#model.to('cuda:0')
#if MU.use_cuda : model.cuda()
evaluate_model(model)

