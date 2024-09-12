'''
load_face_images: This function reads images from a ZIP file containing face images. It iterates over each file in the ZIP archive 
and loads only the JPG images as grayscale using OpenCV's imdecode function. The images are stored in a dictionary where the keys 
are the filenames and the values are the corresponding grayscale image arrays.

display_sample_faces: This function displays a grid of sample face images using Matplotlib. It takes the dictionary of face images 
as input, extracts the last 64 images, and displays them in an 8x8 grid using Matplotlib's subplots function.

extract_person_info: This function extracts information about the person and image number from the filename of a face image. 
It removes the prefix "Grp13Person", splits the filename using "/" as the separator, and extracts the person number and image 
number from the resulting list. The image number is extracted by splitting the filename again using "_" as the separator and taking 
the modulo 10 of the first part.

split_data: This function splits the face images into training and testing datasets. It iterates over each file in the ZIP archive, 
extracts the person and image number using extract_person_info, and assigns the images to either the training or testing dataset 
based on the image number. Images with image numbers 0 and 1 are assigned to the testing dataset, while the rest are assigned to 
the training dataset.

calculate_statistics: This function calculates statistics such as correct prediction percentage, wrong prediction percentage, and 
accuracy based on the number of correct and wrong predictions and the total number of predictions.
'''
import zipfile
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to load face images from a zip file
def load_face_images(zipfile_path):
    faces = {}
    with zipfile.ZipFile(zipfile_path) as facezip:
        for filename in facezip.namelist():
            # Check if the file is a JPG image
            if not filename.endswith(".jpg"):
                continue
            # Read the image from the zip file
            with facezip.open(filename) as image:
                # Decode the image and store it in grayscale
                faces[filename] = cv2.imdecode(np.frombuffer(
                    image.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    return faces

# Function to display a sample of face images
def display_sample_faces(faces):
    # Create a subplot for displaying face images
    fig, axes = plt.subplots(8, 8, sharex=True, sharey=True, figsize=(8, 10))
    # Extract the last 64 face images
    faceimages = list(faces.values())[-64:]
    # Iterate over the face images and display them
    for i in range(64):
        axes[i % 8][i // 8].imshow(faceimages[i], cmap="gray")
    # Display the sample faces
    print("Showing sample faces")
    plt.show()

# Function to extract information about a person from the filename
def extract_person_info(filename):
    # Remove the prefix and split the filename
    filename = filename.replace("Grp13Person", "").split("/")
    # Extract the person number and image number
    person_num = int(filename[0])
    img_num = int(filename[1].split("_")[0]) % 10
    return (person_num, img_num)

# Function to split the data into training and testing sets
def split_data(zipfilepath):
    training_data = {}
    testing_data = {}
    with zipfile.ZipFile(zipfilepath) as facezip:
        for filename in facezip.namelist():
            # Check if the file is a JPG image
            if not filename.endswith(".jpg"):
                continue
            # Extract information about the person from the filename
            person_num, img_num = extract_person_info(filename=filename)
            # Assign images to training or testing set based on the image number
            if img_num == 0 or img_num == 1:
                # Add the image to the testing set
                with facezip.open(filename) as image:
                    testing_data[filename] = cv2.imdecode(np.frombuffer(
                        image.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
            else:
                # Add the image to the training set
                with facezip.open(filename) as image:
                    training_data[filename] = cv2.imdecode(np.frombuffer(
                        image.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    return training_data, testing_data

# Function to calculate prediction statistics
def calculate_statistics(correct_predictions, wrong_predictions, total_predictions):
    # Calculate the percentage of correct and wrong predictions
    correct_percentage = round(correct_predictions / total_predictions, 3)
    wrong_percentage = round(wrong_predictions / total_predictions, 3)
    # Calculate the overall accuracy
    accuracy = round((correct_predictions * 100) / total_predictions, 3)
    return correct_percentage, wrong_percentage, accuracy

