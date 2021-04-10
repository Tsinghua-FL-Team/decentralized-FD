#-----------------------------------------------------------------------------#
#                                                                             #
#   I M P O R T     L I B R A R I E S                                         #
#                                                                             #
#-----------------------------------------------------------------------------#
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
#from torchvision.models import alexnet
from torchvision.models.resnet import ResNet, BasicBlock
import numpy as np
import torchvision

#from functools import partial

#device = 'cuda' if torch.cuda.is_available() else 'cpu'


#*****************************************************************************#
#                                                                             #
#   description:                                                              #
#   create VGG model using the specifications provided.                       #
#                                                                             #
#*****************************************************************************#
class VGG(nn.Module):

    def __init__(self, cfg, size=512, out=10):
        super(VGG, self).__init__()

        self.features = self.make_layers(cfg)
        self.classifier = nn.Sequential(
            #nn.Dropout(),
            nn.Linear(size, size),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(size, size),
            nn.ReLU(True),
            nn.Linear(size, out),
        )

         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


    def make_layers(self, cfg, batch_norm=True):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

def vgg11s():
    return VGG([32, 'M', 64, 'M', 128, 128, 'M', 128, 128, 'M', 128, 128, 'M'], size=128)

def vgg11():
    return VGG([64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'])
  
def vgg16():
    return VGG([64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'])


#*****************************************************************************#
#                                                                             #
#   description:                                                              #
#   create VGG model using the specifications provided.                       #
#                                                                             #
#*****************************************************************************#
class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride, norm_layer):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                norm_layer(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


#*****************************************************************************#
#                                                                             #
#   description:                                                              #
#   create MobileNetV2 model using the specifications provided.               #
#                                                                             #
#*****************************************************************************#
class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)

    def __init__(self, num_classes=10, norm_layer=nn.BatchNorm2d,shrink=1):
        super(MobileNetV2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.norm_layer = norm_layer
        self.cfg = [(1,  16//shrink, 1, 1),
                   (6,  24//shrink, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
                   (6,  32//shrink, 3, 2),
                   (6,  64//shrink, 4, 2),
                   (6,  96//shrink, 3, 1),
                   (6, 160//shrink, 3, 2),
                   (6, 320//shrink, 1, 1)]


        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = self.norm_layer(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(self.cfg[-1][1], 1280//shrink, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = self.norm_layer(1280//shrink)
        self.linear = nn.Linear(1280//shrink, num_classes)


    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride, self.norm_layer))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def mobilenetv2():
    return MobileNetV2(norm_layer=nn.BatchNorm2d)

def mobilenetv2s():
    return MobileNetV2(norm_layer=nn.BatchNorm2d, shrink=2)

def mobilenetv2xs():
    return MobileNetV2(norm_layer=nn.BatchNorm2d, shrink=4)

def mobilenetv2_gn():
    return MobileNetV2(norm_layer=lambda x : nn.GroupNorm(num_groups=2, num_channels=x))
    

#*****************************************************************************#
#                                                                             #
#   description:                                                              #
#   create LENET cifar model using the specifications provided.               #
#                                                                             #
#*****************************************************************************#
class lenet_cifar(nn.Module):
    def __init__(self):
        super(lenet_cifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.binary = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


#*****************************************************************************#
#                                                                             #
#   description:                                                              #
#   create LENET large model using the specifications provided.               #
#                                                                             #
#*****************************************************************************#
class lenet_large(nn.Module):
    def __init__(self):
        super(lenet_large, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(20, 50, 5)
        self.fc1 = nn.Linear(50 * 5 * 5, 512)
        self.fc2 = nn.Linear(512, 10)

    def f(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 50 * 5 * 5)
        x = F.relu(self.fc1(x))
        return x

    def forward(self, x):
        x = self.f(x)
        x = self.fc2(x)
        return x


#*****************************************************************************#
#                                                                             #
#   description:                                                              #
#   create LENET mnist model using the specifications provided.               #
#                                                                             #
#*****************************************************************************#
class lenet_mnist(torch.nn.Module):
    def __init__(self):
        super(lenet_mnist, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)
        self.binary = torch.nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


#*****************************************************************************#
#                                                                             #
#   description:                                                              #
#   function to apply group normalization.                                    #
#                                                                             #
#*****************************************************************************#
def apply_gn(model):
    for n, c in model.named_children():

        
        if isinstance(c, nn.Sequential) or \
                isinstance(c, torch.nn.modules.container.Sequential) or \
                isinstance(c, torchvision.models.resnet.BasicBlock):
            #print("-->", n)
            apply_gn(c)
            
        if isinstance(c, nn.BatchNorm2d):
            #print(n, c.num_features)
            setattr(model, n, torch.nn.GroupNorm(num_groups=4, num_channels=c.num_features))  


#*****************************************************************************#
#                                                                             #
#   description:                                                              #
#   create ResNet models using the specifications provided.                   #
#                                                                             #
#*****************************************************************************#
def resnet8():
    return ResNet(BasicBlock, [1,1,1,1], num_classes=10)


def resnet18():
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)


class Model(nn.Module):
    def __init__(self, feature_dim=128, group_norm=False):
        super(Model, self).__init__()

        self.f = []
        for name, module in resnet8().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(512, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

        if group_norm:
            apply_gn(self)

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)


class resnet8_bn(nn.Module):
    def __init__(self, num_class=10, pretrained_path=None, group_norm=False):
        super(resnet8_bn, self).__init__()

        # encoder
        self.f = Model(group_norm=group_norm).f
        # classifier
        self.fc = nn.Linear(512, num_class, bias=True)


        if pretrained_path:
            self.load_state_dict(torch.load(pretrained_path, map_location='cpu'), strict=False)

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.fc(feature)
        return out


class resnet8_gn(nn.Module):
    def __init__(self, num_class=10, pretrained_path=None, group_norm=True):
        super(resnet8_gn, self).__init__()

        # encoder
        self.f = Model(group_norm=group_norm).f
        # classifier
        self.fc = nn.Linear(512, num_class, bias=True)
        if pretrained_path:
            self.load_state_dict(torch.load(pretrained_path, map_location='cpu'), strict=False)

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.fc(feature)
        return out


#*****************************************************************************#
#                                                                             #
#   description:                                                              #
#   create AlexNet model using the specifications provided.                   #
#                                                                             #
#*****************************************************************************#
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            #nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            #nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return x


class simclrVGG11(nn.Module):
    def __init__(self, n_classes=10, group_norm=False):
        super(simclrVGG11, self).__init__()


        self.f = self.make_layers([64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'])
        # projection head
        self.fc = nn.Linear(512, n_classes, bias=True)#nn.Sequential(nn.Linear(512, 128, bias=False), nn.ReLU(inplace=True), nn.Linear(128, n_classes, bias=True))

         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
                m.bias.data.zero_()

        if group_norm:
            apply_gn(self)


    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.fc(feature)
        return out



class outlier_net(nn.Module):
    def __init__(self):
        super(outlier_net, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 5, bias=False)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 4, 5, bias=False)
        self.fc1 = nn.Linear(4 * 5 * 5, 32, bias=False)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 4 * 5 * 5)
        x = self.fc1(x)
        return x






class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        self.f = nn.Sequential(nn.Linear(512, 128, bias=True), 
                                nn.LeakyReLU(inplace=True), 
                                nn.Linear(128, 32, bias=True),
                                nn.LeakyReLU(inplace=True),
                               nn.Linear(32, 128, bias=True),
                                nn.LeakyReLU(inplace=True),
                               nn.Linear(128, 512, bias=True))  
        
    def forward(self, x):
        return self.f(x)
    
    def get_ae_loss(self, x):
        x_ae = self.f(x)  

        return torch.sum((x_ae - x) ** 2, dim=1)


#*****************************************************************************#
#                                                                             #
#   description:                                                              #
#   create and return the desired model along with optimizer & hyperparams.   #
#                                                                             #
#*****************************************************************************#
def get_model(model):

  return  { "vgg16" : (vgg16, optim.Adam, {"lr":1e-3}),
            "vgg11s" : (vgg11s, optim.Adam, {"lr":1e-3}),
            "vgg11" : (vgg11, optim.Adam, {"lr":1e-3}),
            "resnet18" : (resnet18, optim.Adam, {"lr":1e-3}),
            "alexnet" : (AlexNet, optim.Adam, {"lr":1e-3}),
            "lenet_cifar" : (lenet_cifar, optim.Adam, {"lr":0.001, 
                                                       "weight_decay":0.0
                                                       }),
            "lenet_large" : (lenet_large, optim.SGD, {"lr":0.01, 
                                                      "momentum":0.9, 
                                                      "weight_decay":0.0
                                                      }),
            "lenet_mnist" : (lenet_mnist, optim.Adam, {"lr":0.001, 
                                                       "weight_decay":0.0
                                                       }),
            "mobilenetv2" : (mobilenetv2, optim.SGD, {"lr":0.01, 
                                                      "momentum":0.9, 
                                                      "weight_decay":5e-4
                                                      }),
            "mobilenetv2s" : (mobilenetv2s, optim.SGD, {"lr":0.01, 
                                                        "momentum":0.9, 
                                                        "weight_decay":5e-4
                                                        }),
            "mobilenetv2xs" : (mobilenetv2xs, optim.SGD, {"lr":0.01, 
                                                          "momentum":0.9, 
                                                          "weight_decay":5e-4
                                                          }),
            "mobilenetv2_gn" : (mobilenetv2_gn, optim.SGD, {"lr":0.01, 
                                                            "momentum":0.9, 
                                                            "weight_decay":5e-4
                                                            }),
            "resnet8_gn" : (resnet8_gn, optim.SGD, {"lr":0.1, 
                                                    "momentum":0.9, 
                                                    "weight_decay":5e-4
                                                    }),
            "resnet8_bn" : (resnet8_bn, optim.Adam, {"lr" : 0.001}),
            "simclr_vgg11" : (simclrVGG11, optim.Adam, {"lr" : 0.001, 
                                                        "weight_decay" :5e-4
                                                        })
          }[model]


def print_model(model):
  n = 0
  print("Model:")
  for key, value in model.named_parameters():
    print(' -', '{:30}'.format(key), list(value.shape), "Requires Grad:", value.requires_grad)
    n += value.numel()
  print("Total number of Parameters: ", n) 
  print()