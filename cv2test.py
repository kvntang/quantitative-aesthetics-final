# import cv2
# import numpy as np



# # Start capturing video from the webcam
# cap = cv2.VideoCapture(0)

# frame_index = 0
# while cap.isOpened():
#     success, opencv_frame = cap.read()
#     if not success:
#         print("Ignoring empty camera frame.")
#         continue

    
#     # Convert the frame from BGR (OpenCV default) to RGB.
#     #rgb_frame = cv2.cvtColor(opencv_frame, cv2.COLOR_BGR2RGB)

#     # Resize the frame if necessary to match the input size of your model.
#     # resized_frame = cv2.resize(opencv_frame, (320, 320))


#     # Display the original image
#     cv2.imshow('MediaPipe Object Detection', opencv_frame)
#     if cv2.waitKey(5) & 0xFF == 27:  # Press 'ESC' to exit
#         break

#     frame_index += 1

# # Release resources
# cap.release()



import cv2
import tkinter as tk
from PIL import Image, ImageTk

def update_image():
    ret, frame = cap.read()
    if ret:
        if grayscale.get():  # Check if grayscale mode is toggled on
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)  # Convert to RGB for Tkinter compatibility
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        label.imgtk = imgtk
        label.configure(image=imgtk)
    label.after(20, update_image)

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Create a tkinter window
window = tk.Tk()
window.title("OpenCV and Tkinter")

# Create a label in the window to hold the images
label = tk.Label(window)
label.pack()

# Create a toggle (Checkbutton) to turn grayscale on and off
grayscale = tk.BooleanVar(value=False)
toggle = tk.Checkbutton(window, text="Grayscale ON/OFF", variable=grayscale)
toggle.pack()

# Start the GUI
update_image()
window.mainloop()

# Release the camera and close the window
cap.release()
