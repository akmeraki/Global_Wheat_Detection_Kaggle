import torch.nn as nn
import torch
import torchvision


# Default Pytorch pre-trained FasterRCNN Model 
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

class FasterRCNN_custom(nn.Module):
    def __init__(self, backbone='Resnet50',num_classes = 2, pre_trained=True):
        super(FasterRCNN_custom, self).__init__()
        
        self.classes = num_classes

        if backbone_name == 'mobilenet_v2':
            self.backbone_model = torchvision.models.mobilenet(pretrained=pre_trained).features
            
        elif backbone == 'Resnet152':
            self.backbone_model = torchvision.models.resnet152(pretrained=pre_trained).features

        elif backbone == 'Resnet101':
            self.backbone_model = torchvision.models.resnet101(pretrained= pre_trained).features
        
        elif backbone == 'Resnet50'
            self.backbone_model = torchvision.models.resnet50(pretrained=pre_trained).features
 
    def forward(self,x):
        FasterRCNN(self.backbone_model, self.classes, 



class DenseNet169_change_avg(nn.Module):
    def __init__(self):
        super(DenseNet169_change_avg, self).__init__()
        self.densenet169 = torchvision.models.densenet169(pretrained=True).features
        self.avgpool = nn.AdaptiveAvgPool2d(1)  
        self.relu = nn.ReLU()
        self.mlp = nn.Linear(1664, 6)
        self.sigmoid = nn.Sigmoid()   

    def forward(self, x):
        x = self.densenet169(x)      
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)

        return x

class DenseNet121_change_avg(nn.Module):
    def __init__(self):
        super(DenseNet121_change_avg, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True).features
        self.avgpool = nn.AdaptiveAvgPool2d(1)  
        self.relu = nn.ReLU()
        self.mlp = nn.Linear(1024, 6)
        self.sigmoid = nn.Sigmoid()   

    def forward(self, x):
        x = self.densenet121(x)      
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(-1, 1024)
        x = self.mlp(x)
        
        return x





if __name__ == "__main__":

    FasterRCNN_custom(backbone='Resnet50',pre_trained=True)


















def fetch_model(modelname, num_channels, num_classes, num_filters):
    if modelname == 'FasterRCNN_pytorch':
        model = resunet(num_channels, num_classes, num_filters)
    elif modelname == 'YOLO_pytorch':
        model = resunet(num_channels, num_classes, num_filters)
    elif modelname == 'SSD_pytorch':
        model = fcn(num_channels, num_classes, num_filters)
    else:
        raise ValueError('Check Model spelling, should be one of resunet, unet, fcn in the config'+\
                         'file!')
    return model