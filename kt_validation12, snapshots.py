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
    x, y, r = map(int, xyr)
    mask = np.zeros(shape=img.shape)

    # Ensure the radius is non-negative
    r = max(0, int(r))

    cv2.circle(mask, (x, y), r, (255, 255, 255), -1)
    ns = np.random.random(img.shape)
    mask = cv2.GaussianBlur(mask, (51, 51), 0)
    noisy = (mask.astype(np.float32) * ns).astype(np.uint8) / 255.0
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
    x, y, r = x_y_r

    # Ensure coordinates and radius are within valid ranges
    x = max(r, min(image_size - r, x))
    y = max(r, min(image_size - r, y))
    r = max(0, min(image_size // 2, r))
    r = max(r, 0)


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

def gradient_descent(img, P, ds, gradient_step_size):
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
        percentage_plus_minus, activation_minus = validate_image(image_w_mask_minus, goal_class_index, device, classifierModel)

        dP[i] = (activation_plus - activation_minus) / ds[i] #gradient for each param

    #gradient step
    for i in range(len(P)): #update with new parameters
        P[i] += dP[i]*gradient_step_size #step size controls how much each param change in each iteration

def spotlight_mask(img, image_size, P):
    mask = draw_circle(image_size, P)
    inverted_mask = 255 - mask
    image_w_mask = overlay_image(img, inverted_mask, image_size)
    return image_w_mask
    

def global_search(img,variation_count,iteration_count,class_index):


    buffer_output = validate_image(img, goal_class_index, device, classifierModel)[1]

    print(f'Initial output: {buffer_output}')

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


        for j in range(variation_count):
            x = np.random.randint(max(0,x_base - rand_x_range), min(x_base + rand_x_range, img.shape[1]))
            y = np.random.randint(max(0,y_base - rand_y_range), min(y_base + rand_y_range, img.shape[0]))
            radius = np.random.randint(max(0,radius_base - rand_radius_range), min(radius_base + rand_radius_range, radius_range))

            params = [x, y, radius]
            modified_img = draw_noisy_mask(img, params)
            output = validate_image(modified_img, goal_class_index, device, classifierModel)[1]

            cv2.putText(modified_img, f'general loop: {i}', (10,200), cv2.FONT_HERSHEY_PLAIN,
                1, (255, 255, 255), 1)
            cv2.imshow('Kevin0', modified_img)
            cv2.waitKey(100)

            #update if bigger
            if(output > buffer_output):
                buffer_output = output
                x_base = x
                y_base = y
                radius_base = radius
                print(f'New best: {buffer_output}')


    return x_base, y_base, radius_base, buffer_output


def grid_search(img, iteration_count, class_index, num_grid_points, radii):
    buffer_output = validate_image(img, goal_class_index, device, classifierModel)[1]
    print(f'Initial output: {buffer_output}')
    print(f'Checking class: {classNames[class_index]}')

    # Define grid points
    grid_x = np.linspace(0, img.shape[1], num_grid_points, endpoint=False).astype(int)
    grid_y = np.linspace(0, img.shape[0], num_grid_points, endpoint=False).astype(int)

    # Best parameters
    best_params = [0, 0, 0]  # x, y, radius

    for i in range(iteration_count):
        print(f'Iteration {i}')

        for x in grid_x:
            for y in grid_y:
                for radius in radii:
                    params = [x, y, radius]
                    modified_img = draw_noisy_mask(img, params)
                    output = validate_image(modified_img, goal_class_index, device, classifierModel)[1]

                    print(f'general iteration ({x},{y}): {output}')

                    cv2.putText(modified_img, f'output: {output}', (10,20), cv2.FONT_HERSHEY_PLAIN,
                1, (255, 255, 255), 1)
                    cv2.imshow('General Looking', modified_img)
                    cv2.waitKey(100)

                    if output < buffer_output:
                        buffer_output = output
                        best_params = [x, y, radius]
                        print(f'New best: {buffer_output}')

    return best_params[0], best_params[1], best_params[2], buffer_output


def update_class_index(x):
    global goal_class_index
    goal_class_index = x

def gradient_descent_with_randomness(img, P, ds, gradient_step_size, randomness_scale, iteration):
    dP = [0.0, 0.0, 0.0]

    for i in range(len(P)):
        Pminus = [p for p in P]
        Pminus[i] -= ds[i]

        Pplus = [p for p in P]
        Pplus[i] += ds[i]

        image_w_mask_plus = draw_noisy_mask(img, Pplus)
        _, activation_plus = validate_image(image_w_mask_plus, goal_class_index, device, classifierModel)

        image_w_mask_minus = draw_noisy_mask(img, Pminus)
        _, activation_minus = validate_image(image_w_mask_minus, goal_class_index, device, classifierModel)

        dP[i] = (activation_plus - activation_minus) / (2 * ds[i])

    for i in range(len(P)):
        P[i] += dP[i] * gradient_step_size


def move_circle_randomly(P, step_size, image_size, radius):
    # Randomly adjust x and y coordinates within the boundaries
    P[0] = np.clip(P[0] + np.random.randint(-step_size, step_size + 1), radius, image_size - radius)
    P[1] = np.clip(P[1] + np.random.randint(-step_size, step_size + 1), radius, image_size - radius)
    return P


def set_new_target(image_size, radius):
    # Set a new target position within the boundaries
    target_x = np.random.randint(0, image_size)
    target_y = np.random.randint(0, image_size)
    return [target_x, target_y]

def move_towards_target(P, target, step_fraction):
    # Move P towards the target by a fraction of the distance
    P[0] += int((target[0] - P[0]) * step_fraction)
    P[1] += int((target[1] - P[1]) * step_fraction)
    return P

def adjust_movement_parameters(activation, min_step, max_step, min_interval, max_interval):
    """Adjust step_fraction and update_target_interval based on activation.
    Activation ranges from -1.5 to 1.5.
    """
    # Normalize activation to a 0-1 range
    normalized_activation = (activation + 1.5) / 3.0  # Maps -1.5 to 1.5 to 0 to 1

    # Linearly interpolate step_fraction and update_target_interval based on normalized activation
    step_fraction = min_step + (max_step - min_step) * (1 - normalized_activation)
    update_interval = min_interval + (max_interval - min_interval) * normalized_activation

    return step_fraction, update_interval

def extract_spotlight_area(image, P):
    x, y, r = P
    h, w, _ = image.shape

    # Calculate the coordinates of the region of interest, ensuring they are within image boundaries
    x1, y1 = max(x - r, 0), max(y - r, 0)
    x2, y2 = min(x + r, w), min(y + r, h)

    # Extract the area inside the circle
    spotlighted_area = image[y1:y2, x1:x2]
    return spotlighted_area

def overlay_snapshot(display_image, snapshot, position, radius):
    x, y = position
    h, w, _ = display_image.shape

    # Calculate the coordinates of the region of interest
    x1, y1 = max(x - radius, 0), max(y - radius, 0)
    x2, y2 = min(x + radius, w), min(y + radius, h)

    # Create a mask for the snapshot
    mask = np.zeros_like(snapshot)
    mask_radius = min(radius, min(snapshot.shape[0]//2, snapshot.shape[1]//2))
    cv2.circle(mask, (mask_radius, mask_radius), mask_radius, (255, 255, 255), -1)

    # Overlay the snapshot onto the display image
    roi = display_image[y1:y2, x1:x2]
    snapshot_masked = cv2.bitwise_and(snapshot, mask)
    snapshot_masked_resized = cv2.resize(snapshot_masked, (x2 - x1, y2 - y1))

    roi[np.where((mask[:, :, 0] != 0) & (x2 - x1 > 0) & (y2 - y1 > 0))] = snapshot_masked_resized[np.where((mask[:, :, 0] != 0) & (x2 - x1 > 0) & (y2 - y1 > 0))]
    display_image[y1:y2, x1:x2] = roi
#########################################################
#########################################################
#########################################################
# Main Code
#########################################################
#########################################################
#########################################################


# Initialize circle and target parameters
goal_class_index = 0  # Example class index
image_size = 224
x, y, r = [100, 100, 25]
P = [x, y, r]


target = set_new_target(image_size, r)

# #range setup
# min_step_fraction = 0.01  # Smaller step when close to target
# max_step_fraction = 0.1   # Larger step when searching
# min_update_interval = 60  # Stay longer in an area if activation is high
# max_update_interval = 10  # Update more frequently if activation is low

#initial vals
step_fraction = 0.05  # Fraction of distance to move per frame
update_target_interval = 30  # Frames until updating the target

buffer_activation = 0.0
mask_activation = 0.0
# Initialize webcam
cap = cv2.VideoCapture(0)
cv2.namedWindow('Smooth Movement')
cv2.createTrackbar('Class', 'Smooth Movement', 0, 1, update_class_index)


# drawing_history_1 = []
# drawing_history_2 = []

spotlight_snapshots_1 = []
spotlight_snapshots_2 = []


frame_index = 0
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Resize and preprocess the frame
    frame = cv2.flip(frame, 1)
    img = cv2.resize(frame, (image_size, image_size))
    img = img.astype(np.float32) / 255.0

    # Move the circle smoothly towards the target
    P = move_towards_target(P, target, step_fraction)

    # Update the target position periodically
    if frame_index % update_target_interval == 0:
        target = set_new_target(image_size, r)

    # Every 5 steps, perform validation
    if frame_index % 2 == 0:
        #get activation before the mask
        percentage, buffer_activation = validate_image(img, goal_class_index, device, classifierModel)

        #activation with mask
        image_w_noisy_mask = draw_noisy_mask(img, P)
        percentage, mask_activation = validate_image(image_w_noisy_mask, goal_class_index, device, classifierModel)
        print(f'Validation at step {frame_index}: Buffer: {buffer_activation} Activation = {mask_activation}')
    


    # Display ###########################################################
    #Resize image and P
    resize_factor = 3
    display_size = image_size * resize_factor
    resized_img = cv2.resize(img, (display_size, display_size))
    scaled_P = [int(coord * resize_factor) for coord in P]

    #Spot Light Display
    resized_img = spotlight_mask(resized_img, display_size, scaled_P)
    resized_img = cv2.circle(resized_img, (int(scaled_P[0]), int(scaled_P[1])), int(scaled_P[2]), (255, 255, 255), 2)

    #Compare if the mask made the activation drop, if it did, the covered up part meant something important
    activation_diff = buffer_activation - mask_activation


    # Display Searching Class
    if (goal_class_index == 0):
        cv2.putText(resized_img, "Looking for 'FORMAL'", (10, 40), cv2.FONT_HERSHEY_PLAIN,
            2, (255, 255, 255), 1)
    elif (goal_class_index == 1):
        cv2.putText(resized_img, "Looking for 'INFORMAL'", (10, 40), cv2.FONT_HERSHEY_PLAIN,
            2, (255, 255, 255), 1)


    # #Draw trace
    # for circle in drawing_history_1:
    #     cv2.circle(resized_img, circle, 8, (0, 255, 0), -1)
    # for circle in drawing_history_2:
    #     cv2.circle(resized_img, circle, 8, (0, 0, 255), -1)
        

    # Overlay the snapshots onto the display frame
    for snapshot, position, radius in spotlight_snapshots_1:
        overlay_snapshot(resized_img, snapshot, position, radius)
    for snapshot, position, radius in spotlight_snapshots_2:
        overlay_snapshot(resized_img, snapshot, position, radius)

    # Draw the circle on the image
    if activation_diff > 0.5: #Positive num suggest mask covered a more important part
        #green
        if (goal_class_index == 0):
            # drawing_history_1.append((scaled_P[0], scaled_P[1]))
             # Extract and store the spotlighted area
            spotlight_area = extract_spotlight_area(resized_img, scaled_P)
            spotlight_snapshots_1.append((spotlight_area, (scaled_P[0], scaled_P[1]), scaled_P[2]))

        elif (goal_class_index == 1):
            # drawing_history_2.append((scaled_P[0], scaled_P[1]))
            # Extract and store the spotlighted area
            spotlight_area = extract_spotlight_area(resized_img, scaled_P)
            spotlight_snapshots_2.append((spotlight_area, (scaled_P[0], scaled_P[1]), scaled_P[2]))


        cv2.circle(resized_img, (scaled_P[0], scaled_P[1]), 10, (0, 255, 0), -1)
        cv2.circle(resized_img, (scaled_P[0], scaled_P[1]), scaled_P[2], (0, 255, 0), 5)
        cv2.putText(resized_img, f'{activation_diff:.6f}', (scaled_P[0] - scaled_P[2], scaled_P[1] + scaled_P[2] + 30), cv2.FONT_HERSHEY_PLAIN,
            2, (0, 255, 0), 1)
    elif 0.0< activation_diff < 0.5: 
        #orange
        # cv2.circle(img, (P[0], P[1]), 10, (0, 165, 255), -1)
        cv2.circle(resized_img, (scaled_P[0], scaled_P[1]), scaled_P[2], (0, 165, 255), 5)
        cv2.putText(resized_img, f'{activation_diff:.6f}', (scaled_P[0] - scaled_P[2], scaled_P[1] + scaled_P[2] + 30), cv2.FONT_HERSHEY_PLAIN,
            2, (0, 165, 255), 1)
    elif activation_diff < 0.0:
        #white
        cv2.circle(resized_img, (scaled_P[0], scaled_P[1]), scaled_P[2], (255, 255, 255), 5)
        cv2.putText(resized_img, f'{activation_diff:.6f}', (scaled_P[0] - scaled_P[2], scaled_P[1] + scaled_P[2] + 30), cv2.FONT_HERSHEY_PLAIN,
            2, (255, 255, 255), 1)
        
    
    
   

    # Display the image
    cv2.imshow('Smooth Movement', resized_img)



    key = cv2.waitKey(5)
    if key & 0xFF == 27:  # Press 'ESC' to exit
        break
    elif key & 0xFF == ord('c'):  # Press 'c' to clear the trace
        drawing_history_1 = []
        drawing_history_2 = []

    frame_index += 1

# Release resources
cap.release()
cv2.destroyAllWindows()