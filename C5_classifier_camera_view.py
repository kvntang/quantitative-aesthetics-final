import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import numpy as np
import os
from PIL import Image
import cv2

from classifier import *
from C1_classifier_config import *

#########################################################Configuration
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

#open the camera and apply the model to each frame and display the result in the window title
cap = cv2.VideoCapture(0)
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Our operations on the frame come here
    image = cv2.resize(frame, (image_size, image_size))
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    image = cv2.medianBlur(image,5)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.Canny(image,120,30)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = Image.fromarray(image)
    image = data_transforms(image)
    image = image.unsqueeze(0)
    image = image.to(device)


    # Display the resulting frame
    with torch.no_grad():
        outputs = classifierModel(image)

        #compute the probabilities
        _, preds = torch.max(outputs, 1)
        probabilities = nn.functional.softmax(outputs, dim=1)
        probabilities = probabilities[0].cpu().numpy()*100

        #get the activation of the last layer
        activa = nn.functional.softmax(outputs, dim=1)[0].cpu().detach().numpy()[preds[0]]


        #get the class name
        className = classNames[preds[0]]
        probability = probabilities[preds[0]]

        cv2.putText(frame, f'{className} {probability:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f'{className} {activa:.2f}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2, cv2.LINE_AA)


        cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
