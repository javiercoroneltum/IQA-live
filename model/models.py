import torchvision.models as models
import torch.nn.functional as F
import torch.nn as nn
import torch
import logging


class ResNet101(nn.Module):

    def __init__(self):
        super(ResNet101, self).__init__()
        self.resnet = nn.Conv2d(1, 1, 1)
        self.resnet.head = models.resnet101(pretrained=True)
        self.resnet.head.fc = nn.Sequential(nn.Linear(2*1024, 1024),
                                            nn.ReLU(inplace=True),
                                            nn.Dropout(),
                                            nn.Linear(1024, 10))
        self.resnet.act = nn.Sigmoid()
        
    def forward(self, US):
        head = self.resnet.head(US)
        out = self.resnet.act(head)
        
        return head    


class AlexNet(nn.Module):

    def __init__(self):
        super(AlexNet, self).__init__()
        self.pretrained = models.alexnet(pretrained=True)
        self.pretrained.classifier[6] = nn.Linear(4096, 10)
        self.pretrained.act = nn.Sigmoid()
        

    def forward(self, x):
        features = self.pretrained.features(x)
        out = self.pretrained.classifier(features.view(features.size(0), -1))
        out = self.pretrained.act(out)
        
        return out


class DenseNet(nn.Module):

    def __init__(self):
        super(DenseNet, self).__init__()
        self.densenet = models.densenet161(pretrained=True, drop_rate=0)
        self.densenet.classifier = nn.Sequential(nn.Linear(2208, 1000), nn.ReLU(inplace=True), nn.Dropout(),nn.Linear(1000, 10))
        self.densenet.act = nn.Sigmoid()

    def forward(self, US):
        out = self.densenet(US)
        out = self.densenet.act(out)
        
        return out


def get_architecture(params):
    architecture = params["architecture"]

    if architecture == 'AlexNet':
        logging.info('Using ' + architecture + ' as architecture')
        return AlexNet

    if architecture == 'DenseNet':
        logging.info('Using ' + architecture + ' as architecture')
        return DenseNet

    if architecture == 'ResNet':
        logging.info('Using ' + architecture + ' as architecture')
        return ResNet101

    else:
        logging.info('Architecture not specified, using default')
        return DenseNet
