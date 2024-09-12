'''
Imports:
cv2: OpenCV library for image processing.
numpy: Library for numerical operations in Python.
timeit: Module to measure the execution time of the code.
zipfile: Module to work with ZIP archives.
matplotlib.pyplot: Library for creating visualizations in Python.
utils: Presumably a custom module that contains utility functions for this project.

Function Definitions:
get_matrix: Function to convert the training data into a matrix format suitable for PCA. It creates a NumPy array where each row represents an image.

show_eigen_face: Function to display an eigenface.

Data Loading and Preprocessing:
Load face images from a ZIP archive using the load_face_images function.
Display sample faces using the display_sample_faces function.
Extract image shape (height and width) and the unique labels (person IDs) from the dataset.
Split the dataset into training and testing sets using the split_data function.

Principal Component Analysis (PCA):
Calculate the mean face by taking the mean across all training images.
Compute the mean-subtracted matrix by subtracting the mean face from each training image.
Calculate the covariance matrix g_t based on the mean-subtracted matrix.
Compute the eigenvalues and eigenvectors of the covariance matrix.
Select a specified number of principal components (n) based on user input.
Obtain the eigenvectors (eigenfaces) corresponding to the selected principal components.
Calculate the weight matrix by projecting the mean-subtracted matrix onto the eigenvectors.

Face Recognition:
Define a function get_best_match to find the best match for a given test image using Euclidean distance.
Iterate over the testing set and perform face recognition for each test image.
Compare the predicted person label with the actual person label to evaluate prediction accuracy.

Evaluation:
Calculate and print the accuracy of face recognition.
Measure and print the total time taken for the entire process.
Overall, this code implements a basic face recognition system using PCA, where eigenfaces are used as the basis for representing faces, and Euclidean distance is used for matching faces. However, there are some issues and missing parts in the code, such as the incomplete show_eigen_face function and potential improvements in the face recognition algorithm.

'''

import cv2
import numpy as np
import timeit
import zipfile
from utils import load_face_images, display_sample_faces, extract_person_info, split_data, calculate_statistics
import matplotlib.pyplot as plt

def get_matrix(training_list, img_height, img_width):
    # Initialize an array to store the images
    img_mat = np.zeros((len(training_list), img_height, img_width), dtype=np.uint8)
    
    i = 0
    # Loop through each image in the training list
    for img in training_list:
        # Convert the image to a NumPy matrix
        mat = np.asmatrix(training_list[img])
        # Store the matrix in the image array
        img_mat[i, :, :] = mat
        i += 1
    print("Matrix Size:", img_mat.shape)
    return img_mat

def show_eigen_face(mean_subtracted, eig_no, new_bases):
    # Function to display the eigenface
    ev = new_bases[:, eig_no:eig_no + 1]
    print(new_bases.shape)
    print((mean_subtracted[0]@new_bases).shape)
    print(ev.shape)
    cv2.imshow("Eigen Face " + str(eig_no),  cv2.resize(np.array((80,50), dtype = np.uint8),(200, 200)))
    cv2.waitKey()

# Path to the dataset ZIP file
zipfile_path = "C:\\Users\\shara\\OneDrive\\Desktop\\Final proj\\Dataset_images.zip"

# Load face images from the ZIP file
faces = load_face_images(zipfile_path)
display_sample_faces(faces=faces)

# Get the shape of the face images
faceshape = list(faces.values())[0].shape
print("Face image Height & Width in Pixels:", faceshape)
img_height, img_width = faceshape

# Extract labels from the filenames
labels = set(filename.split("/")[0] for filename in faces.keys())
print("Number of persons in dataset:", len(labels))
print("Number of images of all the persons in Dataset:", len(faces))

# Split the dataset into training and testing sets
training_set, testing_set = split_data(zipfile_path)

start = timeit.default_timer()

# Convert the training set to a matrix
facematrix = get_matrix(training_set, img_height, img_width)
no_of_images = facematrix.shape[0]

# Calculate the mean face
mean_face = np.mean(facematrix, 0)
mean_subtracted = facematrix - mean_face

# Calculate the covariance matrix
mat_width = facematrix.shape[2]
g_t = np.zeros((mat_width, mat_width))
for i in range(no_of_images):
    temp = np.dot(mean_subtracted[i].T, mean_subtracted[i])
    g_t += temp
g_t /= no_of_images

# Perform eigendecomposition
eig_val, eig_vec = np.linalg.eig(g_t)

# Specify the number of principal components
print("\nEnter the Number of Principle Components: ", end="")
n = int(input())

# Extract the top n eigenfaces
eigfaces = eig_vec[:, 0:n]

# Calculate the weight matrix
weight_matrix = np.dot(facematrix, eigfaces)

# Function to find the best match for a given test image
def get_best_match(img):
    img_mat = testing_set[img]
    distances = []
    for i in range(no_of_images):
        temp_imgs = weight_matrix[i]
        dist = np.linalg.norm(img_mat @ eigfaces - temp_imgs)
        distances += [dist]

    min = np.argmin(distances)
    return(min//8 + 1)

stop = timeit.default_timer()
correct_pred = 0
wrong_pred = 0

# Perform face recognition on the testing set
for query_filename in testing_set:
    person_num, _ = extract_person_info(filename=query_filename)
    best_match_index = get_best_match(query_filename)
    if person_num == best_match_index:
        correct_pred += 1
    else:
        wrong_pred += 1
    
total_pred = correct_pred + wrong_pred

# Calculate statistics
correct_percentage, wrong_percentage, accuracy = calculate_statistics(correct_pred, wrong_pred, total_pred)
print("Correct prediction:", correct_percentage)
print("Wrong prediction:", wrong_percentage)
print("Accuracy:", accuracy, "%")
print("Time Taken:", round(stop - start, 3), "s")
