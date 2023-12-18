import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import copy
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import time
from PIL import Image

from classifier import *
from C1_classifier_config import *


#########################################################Configuration
#1 select how many images to use for the embedding projector in tensorboard
imageCountInGraph = 100

#2 select the size of the image thumbnails in the embedding projector
imageSizeInGraph = 64
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
classifierModel = createClassifierModel(classifierModel, len(classNames), freeze_pretrained_parameters = freeze_pretrained_parameters, use_pretrained = False)
classifierModel.load_state_dict(torch.load(modelSaveFile))
classifierModel.to(device)
classifierModel.eval()


data_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

embedding_transforms = transforms.Compose([
        transforms.Resize(imageSizeInGraph),
        transforms.CenterCrop(imageSizeInGraph),
        transforms.ToTensor()
    ])

#export model graph to tensorboard
writer = SummaryWriter(modelSaveFolder)
writer.add_graph(classifierModel, torch.rand(1, 3, 224, 224).to(device))

#collect the file paths for random imageCountInGraph numbner of images from each class
imagePaths = []
imageLabels = []
for i, className in enumerate(classNames):
    classFolder = os.path.join(imageFolder, className)
    classFiles = os.listdir(classFolder)
    classFiles = [os.path.join(classFolder, x) for x in classFiles]
    classFiles = np.random.choice(classFiles, imageCountInGraph, replace=False)
    imagePaths.extend(classFiles)
    imageLabels.extend([i] * len(classFiles))


#initialize the embedding projector
image_count = len(imagePaths)
embedding = torch.zeros(image_count, len(classNames))
embedding_images = torch.zeros(image_count, 3, imageSizeInGraph, imageSizeInGraph)
embedding_labels = torch.zeros(image_count, 1)


#loop through the dataset and get the latent space for each image
for i, imgPath in enumerate(imagePaths):
    # if not imgPath.lower().endswith(('.jpeg', '.jpg')):
    #     continue

    img = Image.open(imgPath).convert('RGB')

    transformed_img = data_transforms(img)
    transformed_img = transformed_img.unsqueeze(0).to(device)
    latent = classifierModel(transformed_img).flatten()
    embedding[i] = latent

    #resize the image to the size for the embedding projector
    embedding_images[i] = embedding_transforms(img)
    #save the label for the embedding projector
    embedding_labels[i] = imageLabels[i]



#save the embedding projector
writer.add_embedding(embedding, label_img=embedding_images, global_step=0)



writer.close()
