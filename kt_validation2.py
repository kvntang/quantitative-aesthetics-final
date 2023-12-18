import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import os
from PIL import Image
import cv2
from classifier import *
from C1_classifier_config import *

#########################################################
# Configuration
#########################################################
freeze_pretrained_parameters = True
print(f'Using classifier model: {classifierModel}')
#where the model is at
baseFolder = os.path.dirname(os.path.abspath(__file__))
modelSaveFolder = os.path.join(baseFolder, f'models/{modelname}/')
modelSaveFile = os.path.join(modelSaveFolder, 'model_weights.pth')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

classFolders = []
for file in os.listdir(imageFolder):
    if os.path.isdir(os.path.join(imageFolder, file)):
        classFolders.append(file)
classNames = [os.path.splitext(x)[0] for x in classFolders]
# Sort the class names so that the order is always the same
classNames.sort()

print(f'Found {len(classNames)} classes: {classNames}')
# Create the model
classifierModel = createClassifierModel(classifierModel, len(classNames), freeze_pretrained_parameters=freeze_pretrained_parameters, use_pretrained=False)
classifierModel.load_state_dict(torch.load(modelSaveFile))
classifierModel.to(device)
classifierModel.eval()
#########################################################
#########################################################


#########################################################
#Main Code
#########################################################
def validate_images(testFolder, device, classifierModel, classNames):
    image_size = 224
    data_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Apply the model to the test images and display the results on the images
    testFiles = os.listdir(testFolder)
    testFiles = [os.path.join(testFolder, x) for x in testFiles]

    for testFile in testFiles:
        print(f'Processing {testFile}')

        if not os.path.splitext(testFile)[1] in ['.jpg', '.png', '.jpeg']:
            print(f'File {testFile} is not an image file. Skipping...')
            continue
        # Load and preprocess the image
        image = Image.open(testFile).convert('RGB')
        image_for_display = cv2.imread(testFile)
        image_tensor = data_transforms(image)
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(device)


        # Forward pass
        with torch.no_grad():
            output = classifierModel(image_tensor)
            _, preds = torch.max(output, 1)
            preds = preds.cpu().numpy()
            output = output.cpu().numpy()[0]

            # Display the class activation and percentage on the image
            for i in range(len(classNames)):
                # activation_text = f'{classNames[i]} Activation: {output[i]:.2f}'
                percentage_text = f'{classNames[i]} Percentage: {np.exp(output[i]) / np.sum(np.exp(output)) * 100:.2f}%'
                font_size = 0.7
                # cv2.putText(image_for_display, activation_text, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(image_for_display, percentage_text, (10, 60 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 200, 0), 2, cv2.LINE_AA)

        # Display or save the result
        cv2.imshow('Result', image_for_display)
        cv2.waitKey(50)


testFolder = 'output_shapes'        
validate_images(testFolder, device, classifierModel, classNames)
cv2.destroyAllWindows()
