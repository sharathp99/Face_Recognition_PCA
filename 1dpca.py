import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import timeit
from utils import load_face_images, display_sample_faces, extract_person_info, split_data, calculate_statistics

# Load face images dataset from a ZIP file
zipfile_path = "C:\\Users\\shara\\OneDrive\\Desktop\\Final proj\\Dataset_images.zip"
face_images = load_face_images(zipfile_path)

# Display a sample of face images
display_sample_faces(face_images)

# Get the shape of face images
image_shape = list(face_images.values())[0].shape
print("Face image Height & Width in Pixels:", image_shape)

# Extract unique labels from the dataset
labels = set(filename.split("/")[0] for filename in face_images.keys())
print("Number of persons in dataset:", len(labels))
print("Number of images of all the persons in Dataset:", len(face_images))

# Prepare training and testing datasets
training_set, testing_set = split_data(zipfile_path)

# Ask the user to input the number of principal components for PCA
print("\n Enter the Number of Principle Components: ", end="")
n_components = int(input())

# Start measuring time for training
start = timeit.default_timer()

# Prepare the face image matrix for PCA
image_matrix = []
image_labels = []
for key, val in training_set.items():
    image_matrix.append(val.flatten())  # Flatten each face image
    image_labels.append(key.split("/")[0])  # Extract the person label

# Convert the face image matrix to a numpy array
image_matrix = np.array(image_matrix)

# Initialize and fit PCA to compute eigenfaces
pca = PCA().fit(image_matrix)

# Get the first n_components eigenfaces
eigenfaces = pca.components_[:n_components]
print("Shape of eigenfaces matrix:", eigenfaces.shape)

# Display the eigenfaces
fig, axes = plt.subplots(n_components // 5 + (n_components % 5 > 0), min(n_components, 5), sharex=True, sharey=True, figsize=(10, 10))
for i in range(n_components):
    axes[i // 5][i % 5].imshow(eigenfaces[i].reshape(image_shape), cmap="gray")
    axes[i // 5][i % 5].set_title(f"Eigenface {i+1}")
    axes[i // 5][i % 5].axis('off')
print("Showing the eigenfaces")
plt.show()

# Compute the weights of face images with respect to the eigenfaces
weights = eigenfaces @ (image_matrix - pca.mean_).T
print("Shape of the weight matrix:", weights.shape)

# Function to find the best match for a given query face image
def find_best_matches(filename):
    query = face_images[filename].reshape(1, -1)
    query_weight = eigenfaces @ (query - pca.mean_).T

    euclidean_distance = np.linalg.norm(weights - query_weight, axis=0)
    best_match_index = np.argmin(euclidean_distance)
    best_match_label = image_labels[best_match_index]

    # Visualize the query and best match face images
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8, 6))
    person_num, img_num = extract_person_info(filename=filename)

    axes[0].imshow(query.reshape(image_shape), cmap="gray")
    axes[0].set_title("Query - Person " + str(person_num))
    axes[1].imshow(image_matrix[best_match_index].reshape(image_shape), cmap="gray")
    axes[1].set_title("Best match - Person " + str((best_match_index // 8) + 1))
    #plt.show()
    return (((best_match_index // 8) + 1), person_num)

# Stop measuring time for training
stop = timeit.default_timer()
training_time = stop - start

# Initialize variables for evaluation
total_persons = len(labels)
total_test_images = len(testing_set)
correct_predictions = 0
wrong_predictions = 0
total_predictions = 0

# Evaluate the recognition performance using testing set
for key, val in testing_set.items():
    predicted_label, actual_label = find_best_matches(filename=key)
    total_predictions += 1
    if predicted_label == actual_label:
        correct_predictions += 1
    else:
        wrong_predictions += 1

# Calculate time taken for recognition per image
time_per_recognition = training_time / total_test_images

# Calculate recognition statistics
correct_percentage, wrong_percentage, accuracy = calculate_statistics(correct_predictions, wrong_predictions, total_predictions)

# Print the results
print("Total Person:", total_persons)
print("Total Test Images:", total_test_images)
print("Correct prediction:", correct_predictions, "/", total_predictions)
print("Correct prediction percentage:", round(correct_percentage, 3) * 100, "%")
print("Wrong prediction:", wrong_predictions, "/", total_predictions)
print("Wrong prediction percentage:", round(wrong_percentage, 3) * 100 , "%")
print("Accuracy: ", accuracy, "%")
print("Training Time:", round( training_time, 3), "seconds")
print("Time Taken for one recognition:", round(time_per_recognition, 3), "seconds")
