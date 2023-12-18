import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import os
from PIL import Image, ImageDraw
import cv2
from classifier import *
from C1_classifier_config import *
import matplotlib.pyplot as plt
import math

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
#MY FUNCTIONS##############
def draw_noisy_mask(img, xyr):
    x, y, r = map(int,xyr)
    mask = np.zeros(shape=img.shape)
    cv2.circle(mask, (x, y), r, (255, 255, 255), -1)
    ns = np.random.random(img.shape)
    mask = cv2.GaussianBlur(mask, (51, 51), 0)
    noisy = (mask.astype(np.float32) * ns).astype(np.uint8)
    mask  = 1-mask.astype(np.float32)/255.0
    img = img * mask + noisy

    return img

def validate_image(image, goal_class_index, device, classifierModel):
    image_size = 224
    data_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # Load and preprocess the image
    # image = Image.open(input_image_path).convert('RGB')
    # image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image = Image.fromarray((image * 255).astype(np.uint8))
    image_tensor = data_transforms(image)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        output = classifierModel(image_tensor)
        _, preds = torch.max(output, 1)
        preds = preds.cpu().numpy()
        output = output.cpu().numpy()[0]

    # percentage = np.exp(output[goal_class_index]) / np.sum(np.exp(output)) * 100
    percentage = np.exp(output[goal_class_index]) / np.sum(np.exp(output))
    percentage= np.array2string(percentage, precision=3, separator=', ', suppress_small=True)
    percentage = float(percentage)

    activation = output[goal_class_index]

    return percentage, activation

def draw_circle(image_size, x_y_r):
    # Create a blank white image
    image = Image.new("RGB", (image_size, image_size), color=(255, 255, 255, 0))
    draw = ImageDraw.Draw(image)

    x, y, r = x_y_r
    draw.ellipse((x - r, y - r, x + r, y + r), fill="black")

    return np.array(image)

def overlay_image(original_image, generated_image, image_size):
    #STEP 4: overlay
        # Resize input image to match the generated image dimensions
    resized_input_image = cv2.resize(original_image, (image_size, image_size))
        # Create a mask from the black parts of the generated image
    mask = generated_image[:, :, 0] == 0
        # Apply the mask to the input image
    output_image = resized_input_image.copy()
    output_image[mask] = 0

    return output_image

def gradient_descent(P, ds, gradient_step_size):
    #gradient of P,so gradient of each parameter
    dP = [0.0, 0.0, 0.0]

    for i in range(len(P)): #loop thru each param
        v = P[i]

        Pminus = [p for p in P] #make a copy
        Pminus[i] -= ds[i] # minus step

        Pplus = [p for p in P] #make a copy
        Pplus[i] += ds[i] # plus step

        image_w_mask_plus = draw_noisy_mask(img, Pplus)
        percentage_plus, activation_plus = validate_image(image_w_mask_plus, goal_class_index, device, classifierModel)
        image_w_mask_minus = draw_noisy_mask(img, Pminus)
        _, activation_minus = validate_image(image_w_mask_minus, goal_class_index, device, classifierModel)

        dP[i] = (activation_plus - activation_minus) / ds[i] #gradient for each param

    #gradient step
    for i in range(len(P)): #update with new parameters
        P[i] += dP[i]*gradient_step_size #step size controls how much each param change in each iteration

def spotlight_mask(image_size, P):
    mask = draw_circle(image_size, P)
    inverted_mask = 255 - mask
    image_w_mask = overlay_image(img, inverted_mask, image_size)
    cv2.imshow('Kevin', image_w_mask)
    cv2.waitKey(100)



def global_search(img,variation_count,iteration_count,class_index):


    buffer_output = validate_image(img, goal_class_index, device, classifierModel)[1]

    print(f'Initial output: {buffer_output}')

    # check if class_to_check is int and within range. This is
    # if not isinstance(class_index, int):
    #     print("class_to_check must be an integer")
    #     return

    # if class_index < 0 or class_index >= len(buffer_output):
    #     print(f"Index is out of range")
    #     return

    print(f'Checking class: {classNames[class_index]}')

    # start from the center
    x_base = img.shape[1] // 2
    y_base = img.shape[0] // 2

    radius_range = min(img.shape[0], img.shape[1]) * 0.5 # adjust the range of the radius here
    radius_base = int(radius_range * 0.5)

    for i in range(iteration_count):
        print(f'Iteration {i}')
        # each iteration shrinks the random range
        rand_pct = (1-(i * (1/iteration_count)))/2
        rand_x_range = int(img.shape[1] * rand_pct)
        rand_y_range = int(img.shape[0] * rand_pct)
        rand_radius_range = int(radius_range * rand_pct)

        # print(f'Random range: {rand_pct}')

        for j in range(variation_count):
            x = np.random.randint(max(0,x_base - rand_x_range), min(x_base + rand_x_range, img.shape[1]))
            y = np.random.randint(max(0,y_base - rand_y_range), min(y_base + rand_y_range, img.shape[0]))
            radius = np.random.randint(max(0,radius_base - rand_radius_range), min(radius_base + rand_radius_range, radius_range))

            params = [x, y, radius]
            modified_img = draw_noisy_mask(img, params)
            output = validate_image(modified_img, goal_class_index, device, classifierModel)[1]
            # print(output[0],buffer_output[0])
            # print(f'x: {x}, y: {y}, radius: {radius}, output: {output}')
            # plot the x,y,radius
            # Now we are only looking at the formal activation. Could look at other activation

            #update if bigger
            if(output > buffer_output):
                buffer_output = output
                x_base = x
                y_base = y
                radius_base = radius
                print(f'New best: {buffer_output}')
                # plt.imshow(modified_img.astype(np.uint8))
                # plt.show()

    return x_base, y_base, radius_base, buffer_output




#########################################################
#########################################################
#########################################################
# Main Code
#########################################################
#########################################################
#########################################################

input_image_path = 'input/a2.jpg'
goal_class_index = 0 #which class to optimize to


image_size = 224
img = cv2.imread(input_image_path)
img = cv2.resize(img, (image_size,image_size))

# # Extract initial values
# first_pass_result = [150, 150, 20] #starting point
# x,y,r = first_pass_result

#1. GENERAL LOOKING
x, y, r, activation = global_search(img,4, 5, goal_class_index)
print(f'General Looking result: {x}, {y}, {r}')


#2. PRECISE LOOKING
P = [x, y, r] # so far three parameters
ds = [2.0 , 2.0, 2.0] #step distance
iteration_count = 100
gradient_step_size = 5

for i in range(iteration_count):
    gradient_descent(P, ds, gradient_step_size)

    #display xyr with new gradient
    print(f'Precise Looking result #{i}: {P}')
    spotlight_mask(image_size, P)

cv2.destroyAllWindows()





