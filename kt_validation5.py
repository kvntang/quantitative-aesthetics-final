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

    return percentage


def average_2d_vectors(vectors):
    # Assuming vectors is a list of four 2D vectors, each represented as a tuple (x, y)
    
    # Extract x and y components
    x_values, y_values = zip(*vectors)

    # Calculate the mean of x and y components
    mean_x = np.mean(x_values)
    mean_y = np.mean(y_values)

    return mean_x, mean_y

def average_3d_vectors(vectors):
    # Assuming vectors is a list of four 3D vectors, each represented as a tuple (x, y, z)
    
    # Extract x, y, and z components
    x_values, y_values, z_values = zip(*vectors)

    # Calculate the mean of x, y, and z components
    mean_x = np.mean(x_values)
    mean_y = np.mean(y_values)
    mean_z = np.mean(z_values)

    return mean_x, mean_y, mean_z

#########################################################
# Main Code
#########################################################
  
#2nd Pass, slowly dialing in on x,y position and radius
#STEP 1: incoming image
input_image_path = 'input/a2.jpg'
goal_class_index = 0 #which class to optimize to


#STEP 2: optimizing loop
#receiving result from 1st pass
first_pass_result = [150, 150, 20] #starting point

    #The Second Pass#############################################################
# Hyperparameters
iteration_count = 20
moving_distance = 10 #level of jitterness

#draw out new image with first_pass_result
image_size = 224
img = cv2.imread(input_image_path)
img = cv2.resize(img, (image_size,image_size))


image_w_mask = draw_noisy_mask(img, first_pass_result)

# Initialize with first_pass_result
best_xyr = first_pass_result
best_percentage = validate_image(image_w_mask, goal_class_index, device, classifierModel)

# Extract initial values
x, y, r = first_pass_result

for _ in range(iteration_count):
    # Try moving in different directions
    vector_list = []
    
    for dx in [-1, 1]:
        for dy in [1, -1]:
            # Try different radii
            for dr in [0, 5, 10, 20]:

                # Apply changes to current position and radius
                new_x = x + dx * moving_distance
                new_y  = y + dy * moving_distance
                new_r = r + dr

                new_xyr = [new_x, new_y, new_r]


                image_w_mask = draw_noisy_mask(img, new_xyr)
                percentage = validate_image(image_w_mask, goal_class_index, device, classifierModel)
                
                vector = [new_x * percentage, new_y * percentage, new_r * percentage]
                vector_list.append(vector)

                # #Save the coordinate if got better result
                # if percentage > best_percentage:
                #     best_xyr = (new_x, new_y, new_r)
                #     best_percentage = percentage
                
                print(new_xyr)
                # 5. Display it (every new generation)
                # cv2.putText(image_w_mask, percentage, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                # cv2.imshow('Kevin', image_w_mask.astype(np.uint8)) 
                # cv2.waitKey(1)


    #average vectors of iteration
    sum_x = 0
    sum_y = 0
    sum_r = 0
    for vector in vector_list:
        x, y, r = vector

        sum_x += x
        sum_y += y
        sum_r += r
    mean_x = sum_x / len(vector_list)
    mean_y = sum_y / len(vector_list)
    mean_r = sum_r / len(vector_list)
        

    new_vector = [mean_x, mean_y, mean_r]


    # x, y , r = x + new_vector[0], y + new_vector[1], new_vector[2] #refresh top value
    x, y , r =new_vector[0], new_vector[1], new_vector[2] #refresh top value

    x = int(x)
    y = int(y)
    r = int(r)

    new_xyr = [x, y, r]

    first_pass_result = best_xyr
    mask = draw_circle(image_size, new_xyr)
    inverted_mask = 255 - mask
    image_w_mask = overlay_image(img, inverted_mask, image_size)
    cv2.putText(image_w_mask, f'Best New Position {best_percentage}%', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Kevin', image_w_mask) 
    cv2.waitKey(100)


# Update current position and radius for the next iteration
   #display final optimized image


##################################################################################################################



#End: loop STEP2 again
cv2.destroyAllWindows()







# print(percentage)










