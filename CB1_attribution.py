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
#1. select the file to process
image_file = 'a3.jpg'

#2. select the folder to store the results to
output_folder = 'output'

#3. select the prefix for the generated files
output_prefix = 'test1'

#4. How many steps to take. Each steps involves erasing a random rectangle and computing the difference between the output of the classifier with and without the rectangle, then adding the difference to the heatmap. More steps will result in a smoother heatmap but will take longer to compute
steps = 2000

#5. select the min/max size of the rectangle to erase. The size is in pixels. The rectangle will be centered at a random location and the size will be randomly selected between min_rect_size and max_rect_size. Using larger rectangles will blur out the fine details of the attribution map but if they are too small, the attribution map will be noisy and patchy
min_rect_size = 1
max_rect_size = 25


#6. select erasure strategy. 
# 'random' will fill each rectangle with a randomly selected solid color
# 'noise' will fill the rectangle with random noise
# 'any' will fill the rectangle either with noise or with a solid random color
# You can also specify a specific color like [0.0, 0.0, 0.0] to fill the rectangle with black
erasure_color = [1.0, 1.0, 1.0]
#########################################################End of configuration



use_softmax = False

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
#classifierModel.eval()


data_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


#start attribution study


with torch.no_grad():
    image = Image.open(image_file).convert('RGB')

    #save the original image in the output folder
    image.save(f'{output_folder}{output_prefix}_original.png')

    image_tensor = data_transforms(image)

    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)

    

    num_classes = len(classNames)

    heat_maps = np.zeros((num_classes, image_size, image_size))

    base_vector = classifierModel(image_tensor)

    if use_softmax:
        base_vector = nn.functional.softmax(base_vector, dim = 1)

    x = torch.arange(image_size).view(-1, 1).repeat(1, image_size).to(device)
    y = torch.arange(image_size).view(1, -1).repeat(image_size, 1).to(device)

 
    if erasure_color == 'any':
        use_random_erasure = True
    else:
        use_random_erasure = False

    for i in range(steps):
        print(f'processing step {i}/{steps}')
        image_copy = image_tensor.clone()

        if use_random_erasure:
            ri = np.random.randint(0, 2)
            if ri == 0:
                erasure_color = 'random'
            elif ri == 1:
                erasure_color = 'noise'


        center_x = np.random.randint(0, image_size - 1)
        center_y = np.random.randint(0, image_size - 1) 
        
        #randomly select the size of the rectangle
        rect_width = np.random.randint(min_rect_size, max_rect_size)
        rect_height = np.random.randint(min_rect_size, max_rect_size)

        x0 = center_x - rect_width // 2
        y0 = center_y - rect_height // 2

        x1 = x0 + rect_width
        y1 = y0 + rect_height

        #make sure the rectangle is inside the image
        x0 = max(x0, 0)
        y0 = max(y0, 0)

        x1 = min(x1, image_size - 1)
        y1 = min(y1, image_size - 1)

        #fill the rectangle with random noise
        image_copy = image_tensor.clone()

        if erasure_color == 'random':
            col = np.random.rand(3)*2.0 - 1.0
            image_copy[0, :, x0:x1, y0:y1] = torch.tensor(col).view(3, 1, 1).to(device)
        elif erasure_color == 'noise':
            image_copy[0, :, x0:x1, y0:y1] = (torch.rand((3, x1-x0, y1-y0))*2.0 - 1.0).to(device)
        else:
            image_copy[0, :, x0:x1, y0:y1] = torch.tensor(erasure_color).view(3, 1, 1).to(device)
        
        #compute the output of the classifier
        vector = classifierModel(image_copy)

        if use_softmax:
            vector = nn.functional.softmax(vector, dim = 1)

        vec_diff = vector - base_vector

        for j in range(num_classes):
            heat_maps[j, x0:x1, y0:y1] += vec_diff[0, j].item()

    
    #normalize the heatmaps
    max_abs = np.max(np.abs(heat_maps))
    heat_maps = heat_maps / max_abs
    
    #save the heatmaps by terning negative values to blue and positive values to red
    pos_color = np.array([255, 0, 0])
    neg_color = np.array([0, 0, 255])

    pos_color = np.expand_dims(pos_color, axis = 0)
    neg_color = np.expand_dims(neg_color, axis = 0)


    for i in range(num_classes):
        heat_map = heat_maps[i]
        rgb_map = np.zeros((image_size, image_size, 3))

        rgb_map[:, :, 0] = heat_map*255
        rgb_map[:, :, 2] = -heat_map*255
        
        rgb_map = np.clip(rgb_map*2.0, 0, 255)

        rgb_map = rgb_map.astype(np.uint8)

        rgb_map = Image.fromarray(rgb_map)

        rgb_map.save(f'{output_folder}{output_prefix}_{classNames[i]}.png')


