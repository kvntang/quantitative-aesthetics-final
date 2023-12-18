import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import copy
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import time

from classifier import *
from C1_classifier_config import *

#########################################################Configuration
#1 set the number of epochs to train the model for
num_epochs = 8

#########################################################End of configuration

print(f'Using classifier model: {classifierModel}')

baseFolder = os.path.dirname(os.path.abspath(__file__))
modelSaveFolder = os.path.join(baseFolder, f'models/{modelname}/')
modelSaveFile = os.path.join(modelSaveFolder, 'model_weights.pth')

image_size = 224
image_channels = 3
freeze_pretrained_parameters = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

classFolders = []
for file in os.listdir(imageFolder):
    if os.path.isdir(os.path.join(imageFolder, file)):
        classFolders.append(file)
classNames = [os.path.splitext(x)[0] for x in classFolders]
#sort the class names so that the order is always the same
classNames.sort()

print(f'Found {len(classNames)} classes: {classNames}')

# Create the model
classifierModel = createClassifierModel(classifierModel, len(classNames), freeze_pretrained_parameters = freeze_pretrained_parameters, use_pretrained = True)
classifierModel.to(device)
classifierModel.train()

#initialize the tensorboard logging system
writer = SummaryWriter(modelSaveFolder)

# Data augmentation and normalization for training
data_transforms =transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


print("Initializing Dataset and Dataloader...")
# Create training dataset
image_dataset = datasets.ImageFolder(imageFolder, data_transforms)
# Create training dataloader
dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=4, shuffle=True, num_workers=0)

params_to_update = classifierModel.parameters()
print("Params to learn:")
if freeze_pretrained_parameters:
    params_to_update = []
    for name,param in classifierModel.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in classifierModel.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

# Observe that all parameters are being optimized
optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

# Setup the loss fxn
criterion = nn.CrossEntropyLoss()

# Train loop
since = time.time()

for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    running_loss = 0.0
    running_corrects = 0

    # Iterate over data.
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(True):
            # Get model outputs and calculate loss
            outputs = classifierModel(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

            # backward + optimize
            loss.backward()
            optimizer.step()

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)

    #tensorboard logging
    writer.add_scalar('Loss/train', epoch_loss, epoch)
    writer.add_scalar('Accuracy/train', epoch_acc, epoch)

    print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
    print()

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

writer.flush()

#save the model
if not os.path.exists(modelSaveFolder):
    os.makedirs(modelSaveFolder)
torch.save(classifierModel.state_dict(), modelSaveFile)
