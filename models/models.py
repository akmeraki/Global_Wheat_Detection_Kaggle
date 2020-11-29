import torch.nn as nn
import torch
import torchvision

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
        
        elif backbone == 'Resnet50':
            self.backbone_model = torchvision.models.resnet50(pretrained=pre_trained).features
 

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

def pytorch_FASTER_RCNN(backbone, pre_trained):

    if backbone == 'Resnet50_fpn':
        # load a model pre-trained pre-trained on COCO
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pre_trained)

        # replace the classifier with a new one, that has
        # num_classes which is user-defined
        num_classes = 2  # 1 class (person) + background
        
        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    elif backbone == 'mobilenet':
        # load a pre-trained model for classification and return only the features
        backbone = torchvision.models.mobilenet_v2(pretrained=True).features
        
        # FasterRCNN needs to know the number of output channels in a backbone.
        #  For mobilenet_v2, it's 1280 so we need to add it here
        backbone.out_channels = 1280

        # let's make the RPN generate 5 x 3 anchors per spatial
        # location, with 5 different sizes and 3 different aspect
        # ratios. We have a Tuple[Tuple[int]] because each feature
        # map could potentially have different sizes and
        # aspect ratios
        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                        aspect_ratios=((0.5, 1.0, 2.0),))

        # let's define what are the feature maps that we will
        # use to perform the region of interest cropping, as well as
        # the size of the crop after rescaling.
        # if your backbone returns a Tensor, featmap_names is expected to
        # be [0]. More generally, the backbone should return an
        # OrderedDict[Tensor], and in featmap_names you can choose which
        # feature maps to use.
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                        output_size=7,
                                                        sampling_ratio=2)

        # put the pieces together inside a FasterRCNN model
        model = FasterRCNN(backbone,
                        num_classes=2,
                        rpn_anchor_generator=anchor_generator,
                        box_roi_pool=roi_pooler)

    return model 


# if __name__ == "__main__":

#     model = pytorch_FASTER_RCNN(backbone='Resnet50_fpn', pre_trained=True)
    
















