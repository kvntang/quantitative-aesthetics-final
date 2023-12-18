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
    x, y, r = xyr
    mask = np.zeros(shape=img.shape)
    # print(mask.dtype)
    cv2.circle(mask, (x, y), r, (255, 255, 255), -1)

    # ns = np.random.random((img.shape[0], img.shape[1]))
    ns = np.random.random(img.shape)
    # print(ns.dtype)
    # ns3 = np.dstack([ns, ns, ns])
    # print(mask.dtype)
    mask = cv2.GaussianBlur(mask, (51, 51), 0)
    noisy = (mask.astype(np.float32) * ns).astype(np.uint8)
    mask  = 1-mask.astype(np.float32)/255.0
    img = img * mask + noisy
    # plt.imshow(img.astype(np.uint8))
    # plt.show()
    return img

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


def normalize_vector(v):
    # Calculate the magnitude of the vector
    magnitude = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    
    # Avoid division by zero
    if magnitude == 0:
        return (0, 0, 0)

    # Normalize the vector
    return (v[0] / magnitude, v[1] / magnitude, v[2] / magnitude)




#########################################################
# Main Code
#########################################################

input_image_path = 'input/a2.jpg'
goal_class_index = 0 #which class to optimize to
first_pass_result = [150, 150, 20] #starting point

iteration_count = 50
moving_distance = 5 #level of jitterness

image_size = 224
img = cv2.imread(input_image_path)
img = cv2.resize(img, (image_size,image_size))


# Extract initial values
x = 130
y = 130
r = 40

P = [x, y, r] # so far three parameters
ds = [2.0 , 2.0, 2.0] #step distance

# gradient calculation
for i in range(iteration_count):
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
        percentage_minus, activation_minus = validate_image(image_w_mask_minus, goal_class_index, device, classifierModel)

        dP[i] = (activation_plus - activation_minus) / ds[i] #gradient for each param

    #gradient step
    for i in range(len(P)):
        P[i] += dP[i]*0.01
        #This is done by adding the product of the gradient and a small factor (0.01) to each parameter. This step size (0.01) 
        # controls how much the parameters change in each iteration and is crucial for the convergence of the algorithm.

    #display xyr with new gradient
    image_w_new_gradient = draw_noisy_mask(img, P)
    percentage, activation = validate_image(image_w_new_gradient, goal_class_index, device, classifierModel)
    cv2.putText(image_w_new_gradient, percentage, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Kevin', image_w_new_gradient.astype(np.uint8)) 
    cv2.waitKey(1)

    # mask = draw_circle(image_size, new_xyr)
    # inverted_mask = 255 - mask
    # image_w_mask = overlay_image(img, inverted_mask, image_size)
    # cv2.imshow('Kevin', image_w_mask)
    # cv2.waitKey(100)



    # xyr_pattern = [[-1, 0, 0], [1, 0, 0],
    #            [0, 1, 0], [0, -1]]
    
    # r_pattern = [-1, 0, 1]
    
    # for dx, dy  in xy_pattern:
    #     for dr in r_pattern:
    #         # new coordinates
    #         new_x = x + dx * moving_distance
    #         new_y  = y + dy * moving_distance
    #         new_r = r + dr

            # new_xyr = [new_x, new_y, new_r]

            #  #masked image
            # image_w_mask = draw_noisy_mask(img, new_xyr)
            # percentage, activation = validate_image(image_w_mask, goal_class_index, device, classifierModel)
            
            # # new vector
            # vector = [dx * activation, dy * activation, dr * activation]
            # vector_list.append(vector)

            # cv2.putText(image_w_mask, percentage, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            # cv2.imshow('Kevin', image_w_mask.astype(np.uint8)) 
            # cv2.waitKey(1)




    #average vectors of iteration
    # sum_x = 0
    # sum_y = 0
    # sum_r = 0
    # for vector in vector_list:
    #     vx, vy, vr = vector

    #     sum_x += vx
    #     sum_y += vy
    #     sum_r += vr

    # mean_x = sum_x / len(vector_list)
    # mean_y = sum_y / len(vector_list)
    # mean_r = sum_r / len(vector_list)
    
    # new_vector = [mean_x, mean_y, mean_r] #floats

    # print(f'new vector{new_vector}')
    # print (f'old coord{x, y, r}')

    # x, y , r = x + x * new_vector[0], y + y * new_vector[1], r + r * new_vector[2] #refresh top value
    # x = int(x)
    # y = int(y)
    # r = int(r)
    # new_xyr = [x, y, r] # ints
    # print(f'new coord{new_xyr}')

    # mask = draw_circle(image_size, new_xyr)
    # inverted_mask = 255 - mask
    # image_w_mask = overlay_image(img, inverted_mask, image_size)
    # cv2.imshow('Kevin', image_w_mask)
    # cv2.waitKey(100)


cv2.destroyAllWindows()





