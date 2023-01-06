import xml.etree.ElementTree as ET
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from PIL import Image
import os

# Define a function to extract features from the images
def extract_features(image):
    # Convert the image to a NumPy array with a consistent shape
    image_array = np.array(image, dtype=np.float32)
    image_array = image_array.reshape((1,) + image_array.shape)
    
    # Extract the features from the image
    features = np.array(image_array).flatten()
    return features


# Define a function to extract labels from the annotations
def extract_labels(annotation):
    labels = []
    
    # Iterate over the objects in the annotation
    for obj in annotation.findall("object"):
        # Extract the label (i.e., the text in the image)
        label = obj.find("name").text
        labels.append(label)

    return labels


# Define a function to evaluate the accuracy of the model
def evaluate_accuracy(predictions, labels):
    # Calculate the number of correct predictions
    correct_predictions = sum(predictions == labels)
    
    # Calculate the total number of predictions
    total_predictions = len(predictions)
    
    # Calculate the accuracy as a percentage
    accuracy = correct_predictions / total_predictions * 100
    
    return accuracy

def extract_data():
    # Path to the annotations folder
    annotation_folder = "./assets/annotations"

    # List to store the features and labels
    X = []
    y = []

    # Iterate over the annotations in the folder
    for file in os.listdir(annotation_folder):
        # Ignore hidden files
        if file.startswith("."):
            continue
        
        # Construct the full path to the annotation file
        file_path = os.path.join(annotation_folder, file)
        
        # Parse the XML tree from the file
        tree = ET.parse(file_path)

        # Extract the labels from the annotation file
        labels = extract_labels(tree)

        # Add labels to y
        y.extend(labels)

        # Find the filename of the image
        image_filename = tree.find("filename").text
        
        # Construct the full path to the image file
        image_path = os.path.join("./assets/labeled-images", image_filename)
        
        # Open the image
        image = Image.open(image_path)
        
        # Extract the width and height of the image
        width = int(tree.find("size/width").text)
        height = int(tree.find("size/height").text)
        
        # Iterate over the objects in the annotation
        for obj in tree.findall("object"):
            # Extract the label (i.e., the text in the image)
            label = obj.find("name").text
            
            # Extract the bounding box coordinates
            xmin = int(round(float(obj.find("bndbox/xmin").text)))
            ymin = int(round(float(obj.find("bndbox/ymin").text)))
            xmax = int(round(float(obj.find("bndbox/xmax").text)))
            ymax = int(round(float(obj.find("bndbox/ymax").text)))
            
            # Crop the image to the bounding box
            cropped_image = image.crop((xmin, ymin, xmax, ymax))
            
            # Resize the image to a fixed size
            resized_image = cropped_image.resize((200, 200))

            # Convert the image to a NumPy array with a consistent shape
            image_array = np.array(resized_image, dtype=np.float32)
            image_array = image_array.reshape((1,) + image_array.shape)

            # Extract features from the image
            features = extract_features(image_array)


            # Add the features to the list
            X.append(features)

    return X, y


def train_model(X, y):
    # Convert the lists to numpy arrays
    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    # Make sure the images have a consistent shape
    X = X.reshape(X.shape[0], -1)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Create an SVM classifier
    clf = SVC()

    # Train the classifier on the training data
    clf.fit(X_train, y_train)

    # Use the classifier to make predictions on the test data
    predictions = clf.predict(X_test)

    # Evaluate the performance of the classifier
    accuracy = evaluate_accuracy(predictions, y_test)

    print(f"Accuracy: {accuracy:.2f}")

# Convert extract_data from tuple to two variables
X, y = extract_data()
train_model(X, y)


