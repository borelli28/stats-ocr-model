import xml.etree.ElementTree as ET
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from PIL import Image
import os
import matplotlib.pyplot as plt


# Extract features from the images
def extract_features(image):
    print("Extracting features...")
    # Convert the image to a NumPy array with a consistent shape
    image_array = np.array(image, dtype=np.float32)
    image_array = image_array.reshape((1,) + image_array.shape)
    
    features = np.array(image_array).flatten()

    return features


# Define a function to extract labels from the annotations
def extract_labels(annotation):
    print("Extracting labels...")
    labels = []
    
    # Iterate over the objects in the annotation
    for obj in annotation.findall("object"):
        # Extract the label (i.e., the text in the image)
        label = obj.find("name").text
        labels.append(label)

    return labels


# Define a function to evaluate the accuracy of the model
def evaluate_accuracy(predictions, labels):
    print("Evaluating accuracy...")
    # Calculate the number of correct predictions
    correct_predictions = sum(predictions == labels)
    
    # Calculate the total number of predictions
    total_predictions = len(predictions)
    
    # Calculate the accuracy as a percentage
    accuracy = correct_predictions / total_predictions * 100
    
    return accuracy


def extract_data():
    print("Extracting data from training data...")
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
        
        # Iterate over the objects in the annotation
        for obj in tree.findall("object"):
            
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


# Plot the training and validation loss over time
def plot_loss(X_train, y_train, X_val, y_val, C_values):
    # Create lists to store the training and validation loss
    train_loss = []
    val_loss = []

    # Train the model with different values of the regularization parameter C
    for C in C_values:
        # Create an SVM model with the given value of C
        svm = SVC(C=C, kernel='linear')

        # Train the model on the training data
        svm.fit(X_train, y_train)

        # Calculate the training and validation loss
        train_loss.append(svm.score(X_train, y_train))
        val_loss.append(svm.score(X_val, y_val))

    # Plot the training and validation loss
    plt.plot(C_values, train_loss, 'bo', label='Training loss')
    plt.plot(C_values, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Regularization parameter C')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()


def train_model(X, y):
    print("Training model...")
    # Convert the lists to numpy arrays
    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    # Make sure the images have a consistent shape
    X = X.reshape(X.shape[0], -1)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    # Create an SVM classifier
    clf = SVC()

    # Train the classifier on the training data
    clf.fit(X_train, y_train)

    # Use the classifier to make predictions on the test data
    predictions = clf.predict(X_test)

    # Evaluate the performance of the classifier
    accuracy = evaluate_accuracy(predictions, y_test)

    print(f"Accuracy: {accuracy:.2f}")

    print("\n*Test*")
    print(y_test)
    print("--------------------------------------------------")
    print(predictions)
    print("*Prediction*\n")

    # Prints count of labels and features
    len_y = len(y)
    print(f"y(labels) length: {len_y}".format(len_y))
    len_x = len(X)
    print(f"X(features) length: {len_x}".format(len_x))

    # Plot the loss for different values of C
    # plot_loss(X_train, y_train, X_test, y_test, [0.01, 0.1, 1, 10, 100])
    """
    Training loss is a measure of how well the model is able to fit
    the training data.
    It is calculated as the average of the errors made by the model 
    on the training data.

    Validation loss is a measure of how well the model is able to 
    generalize to new, unseen data.
    It is calculated as the average of the errors made 
    by the model on the validation data.
    """


def run():
    # Convert extract_data from tuple to variables
    X, y = extract_data()
    train_model(X, y)


run()
