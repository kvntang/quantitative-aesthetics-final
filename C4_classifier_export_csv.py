import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import numpy as np
import os
from PIL import Image
from classifier import *

from classifier import *
from C1_classifier_config import *

#########################################################Configuration
#1 select the folder to load the images from
testFolder = 'data/validation_set_processed'

#2 select the csv file to save the classification results
csvFile = 'classified.csv'
#########################################################End of configuration


print(f'Using classifier model: {classifierModel}')

baseFolder = os.path.dirname(os.path.abspath(__file__))
modelSaveFolder = os.path.join(baseFolder, f'models/{modelname}/')
modelSaveFile = os.path.join(modelSaveFolder, 'model_weights.pth')


#set csv file in th ecurrent python file folder
csvFile = os.path.join(os.path.dirname(os.path.realpath(__file__)), csvFile)



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

#apply the model to the test images and save the results in a csv file with the image name, the predicted class and the probability of the prediction
testFiles = os.listdir(testFolder)
testFiles = [os.path.join(testFolder, x) for x in testFiles]

with open(csvFile, 'w') as f:
    f.write('image')
    for className in classNames:
        f.write(f', {className} activation')

    for className in classNames:
        f.write(f', {className}%')

    f.write('\n')

    for testFile in testFiles:
        f.write(f'{testFile}')
        print(f'processing {testFile}')

        if not os.path.splitext(testFile)[1] in ['.jpg', '.png', '.jpeg']:
            print(f'File {testFile} is not an image file. Skipping...')
            continue
        image = Image.open(testFile).convert('RGB')
        image = data_transforms(image)
        image = image.unsqueeze(0)
        image = image.to(device)
        with torch.no_grad():
            output = classifierModel(image)
            _, preds = torch.max(output, 1)
            preds = preds.cpu().numpy()
            output = output.cpu().numpy()[0]

            for i in range(len(classNames)):
                f.write(f', {output[i]}')

            output = (np.exp(output) / np.sum(np.exp(output)))*100

            for i in range(len(classNames)):
                f.write(f', {output[i]}')

        f.write('\n')
