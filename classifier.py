import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def set_parameter_requires_grad(model, freeze_pretrained_parameters):
    if freeze_pretrained_parameters:
        for param in model.parameters():
            param.requires_grad = False


def createClassifierModel(model_name, num_classes, freeze_pretrained_parameters, use_pretrained=True) -> nn.Module:
    model_ft = None

    if model_name == "resnet":
        """ Resnet18
        """

        if use_pretrained:
            model_ft = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        else:
            model_ft = models.resnet18()

        set_parameter_requires_grad(model_ft, freeze_pretrained_parameters)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == "alexnet":
        """ Alexnet
        """

        if use_pretrained:
            model_ft = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
        else:
            model_ft = models.alexnet()

        set_parameter_requires_grad(model_ft, freeze_pretrained_parameters)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
    elif model_name == "vgg":
        """ VGG11_bn
        """

        if use_pretrained:
            model_ft = models.vgg11_bn(weights=models.VGG11_BN_Weights.DEFAULT)
        else:
            model_ft = models.vgg11_bn()
            
        set_parameter_requires_grad(model_ft, freeze_pretrained_parameters)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
    elif model_name == "squeezenet":
        """ Squeezenet
        """

        if use_pretrained:
            model_ft = models.squeezenet1_0(weights=models.SqueezeNet_Weights.DEFAULT)
        else:
            model_ft = models.squeezenet1_0()
            
        set_parameter_requires_grad(model_ft, freeze_pretrained_parameters)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
    elif model_name == "densenet":
        """ Densenet
        """

        if use_pretrained:
            model_ft = models.densenet121(weights=models.DenseNet_Weights.DEFAULT)
        else:
            model_ft = models.densenet121()


        set_parameter_requires_grad(model_ft, freeze_pretrained_parameters)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
    elif model_name == "vit_b_32":
        """ Vision Transformer B_32 (ViT-B_32) """
        if use_pretrained:
            model_ft = models.vit_b_32(weights=models.ViT_B_32_Weights.DEFAULT)
        else:
            model_ft = models.vit_b_32()

        set_parameter_requires_grad(model_ft, freeze_pretrained_parameters)
        num_ftrs = model_ft.heads.head.in_features
        model_ft.heads.head = nn.Linear(num_ftrs, num_classes)
    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft