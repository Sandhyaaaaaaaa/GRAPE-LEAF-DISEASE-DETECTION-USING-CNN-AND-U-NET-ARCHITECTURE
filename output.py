
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from PIL import Image, ImageTk
import cv2
import tkinter as tk
from tkinter import filedialog

# Set up file paths
data_dir = r"D:\\grape leaf disease detection\\grape leaf disease detection"
disease_folders = ['Black rot', 'Black measles', 'Grape healthy', 'Leaf blight']

# Set image size
img_size = (224, 224)

# Define the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(disease_folders), activation='softmax'))  # Output layer with softmax for multi-class classification

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Load the trained model
model.load_weights(r"D:\grape leaf disease detection\grape_leaf_disease_model.weights.h5")


# Function to predict the image
def predict_image(image_path):
    img = Image.open(image_path).convert('L')
    img = img.resize(img_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return prediction[0]


# Function to perform segmentation and annotate the image
def segment_and_annotate(image_path):
    original_img = cv2.imread(image_path)
    img = original_img.copy()

    # Perform segmentation using OpenCV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Adjust the color ranges to focus on brownish infected parts
    lower_brown = np.array([10, 100, 50])
    upper_brown = np.array([30, 255, 200])
    mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)

    # Create the final mask
    mask = mask_brown

    # Mark the infected areas with a color
    marked_img = original_img.copy()
    marked_img[mask > 0] = (0, 255, 0)  # Green color

    return original_img, marked_img


# GUI function
def run_gui():
    def select_image():
        global image_path
        image_path = filedialog.askopenfilename(initialdir=".", title="Select an image",
                                                filetypes=(("Image files", "*.jpg;*.jpeg;*.png"), ("all files", "*.*")))
        if image_path:
            original_img, marked_img = segment_and_annotate(image_path)

            # Display the original image
            original_photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)))
            original_label.configure(image=original_photo)
            original_label.image = original_photo

            # Display the marked image
            marked_photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(marked_img, cv2.COLOR_BGR2RGB)))
            marked_label.configure(image=marked_photo)
            marked_label.image = marked_photo

            # Predict the image
            predictions = predict_image(image_path)
            max_index = np.argmax(predictions)
            if max_index >= 0 and max_index < len(disease_folders):
                disease_name = disease_folders[max_index]
                prediction_label.configure(text=f"Predicted disease: {disease_name}")
            else:
                prediction_label.configure(text="No disease predicted.")

    # Create the GUI
    root = tk.Tk()
    root.title("Grape Leaf Disease Detection")

    # Create the image display area
    original_label = tk.Label(root)
    original_label.grid(row=0, column=0, padx=10, pady=10)

    marked_label = tk.Label(root)
    marked_label.grid(row=0, column=1, padx=10, pady=10)

    # Create the prediction label
    prediction_label = tk.Label(root, font=("Arial", 14))
    prediction_label.grid(row=1, column=0, columnspan=2, pady=10)

    # Create the select image button
    select_button = tk.Button(root, text="Select Image", command=select_image)
    select_button.grid(row=2, column=0, columnspan=2, pady=10)

    root.mainloop()


# Run the GUI
run_gui()