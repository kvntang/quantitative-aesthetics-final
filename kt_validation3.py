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
def draw_noisy_mask(img, result):
    x, y, r = result

    mask = np.zeros(shape=img.shape)
    print(mask.dtype)
    cv2.circle(mask, (x, y), r, (255, 255, 255), -1)

    # ns = np.random.random((img.shape[0], img.shape[1]))
    ns = np.random.random(img.shape)
    # print(ns)
    # ns3 = np.dstack([ns, ns, ns])
    # print(mask.dtype)
    mask = cv2.GaussianBlur(mask, (51, 51), 0)
    noisy = (mask.astype(np.float32) * ns).astype(np.uint8)
    mask = 1-mask.astype(np.float32)/255.0
    # img = img * mask + noisy
    plt.imshow(img.astype(np.uint8))
    plt.show()

    draw_noisy_mask(img, 100, 100, 50)


def draw_circle(image_size, x_y_r):
    # Create a blank white image
    image = Image.new("RGB", (image_size, image_size), color=(255, 255, 255, 0))
    draw = ImageDraw.Draw(image)

    x, y, r = x_y_r
    draw.ellipse((x - r, y - r, x + r, y + r), fill="black")

    return np.array(image)



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
    image_tensor = data_transforms(image)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)

    # Forward pass
    with torch.no_grad():
        output = classifierModel(image_tensor)
        _, preds = torch.max(output, 1)
        preds = preds.cpu().numpy()
        output = output.cpu().numpy()[0]

    # percentage = np.exp(output[goal_class_index]) / np.sum(np.exp(output)) * 100
    percentage = np.exp(output[goal_class_index]) / np.sum(np.exp(output)) * 100
    percentage= np.array2string(percentage, precision=10, separator=', ', suppress_small=True)
    return percentage





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
iteration_count = 1
moving_distance = 6 #level of jitterness

# Extract initial values
x, y, r = first_pass_result

# Initialize with first_pass_result
best_result_xyr = first_pass_result
#OG image + generate image ==> validate
best_percentage = validate_image(img, x, y, r)


for _ in range(iteration_count):
    # Try moving in different directions
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            # Try different radii
            for dr in [-1, 0, 1]:

                # Apply changes to current position and radius
                new_x = x + (dx * moving_distance)
                new_y  = y + (dy * moving_distance)
                new_r = r + dr

            
                # 1.Generate new circle
                # 2.Overlay it 
                # 3 Validate it
                # 4. Save the coordinate if got better result
                # 5. Display it

                # Update the best result if needed
                if percentage > best_percentage:
                    best_result = (new_x, new_y, new_r)
                    best_percentage = percentage

    # Update current position and radius for the next iteration
    x, y, r = best_result

##################################################################################################################

#STEP 3: draw out new image
img = cv2.imread(input_image_path)
image_size = 300
generated_image = draw_circle(image_size, first_pass_result)

#STEP 4: overlay
    # Resize input image to match the generated image dimensions
resized_input_image = cv2.resize(img, (image_size, image_size))
    # Create a mask from the black parts of the generated image
mask = generated_image[:, :, 0] == 0
    # Apply the mask to the input image
output_image = resized_input_image.copy()
output_image[mask] = 0

    #display validation#########################################################
    #turn output_image into a PIL image for validation
output_image_pil = Image.fromarray(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
percentage = validate_image(output_image_pil, goal_class_index, device, classifierModel)
cv2.putText(output_image, percentage, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    #######################################################
    # Display the output image
cv2.imshow('Result', output_image)
cv2.waitKey(5000)

#End: loop STEP2 again
cv2.destroyAllWindows()







# print(percentage)










