import xml.etree.ElementTree as ET
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from PIL import Image
import os
import matplotlib.pyplot as plt
from heapq import nlargest
import pickle


def extract_features(image):
    # Convert the image to a NumPy array with a consistent shape
    image_array = np.array(image, dtype=np.float32)
    image_array = image_array.reshape((1,) + image_array.shape)
    
    features = np.array(image_array).flatten()

    return features


def extract_labels(annotation):
    labels = []
    
    # Iterate over the objects in the annotation
    for obj in annotation.findall("object"):
        # Extract the label (i.e., the text in the image)
        label = obj.find("name").text
        labels.append(label)

    return labels


def evaluate_accuracy(predictions, labels):
    print("Evaluating accuracy...")
    correct_predictions = sum(predictions == labels)
    total_predictions = len(predictions)
    accuracy = correct_predictions / total_predictions * 100
    
    return accuracy


def extract_data():
    print("Extracting data from training data...")
    annotation_folder = "../assets/annotations"

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
        # if the filename does not have an extension, add one
        if image_filename[-1] != "g":
            image_filename = image_filename + ".png"
        
        # Construct the full path to the image file
        image_path = os.path.join("../assets/labeled-images", image_filename)
        
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

            features = extract_features(image_array)

            # Add the features to the list
            X.append(features)

    return X, y


# Plot the training and validation loss over time
def plot_loss(X_train, y_train, X_val, y_val, C_values):
    train_loss = []
    val_loss = []

    # Train the model with different values of the regularization parameter C
    for C in C_values:
        # Create an SVM model with the given value of C
        svm = SVC(C=C, kernel="linear")

        # Train the model on the training data
        svm.fit(X_train, y_train)

        # Calculate the training and validation loss
        train_loss.append(svm.score(X_train, y_train))
        val_loss.append(svm.score(X_val, y_val))

    # Plot the training and validation loss
    plt.plot(C_values, train_loss, "bo", label="Training loss")
    plt.plot(C_values, val_loss, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.xlabel("Regularization parameter C")
    plt.ylabel("Loss")
    plt.legend()

    plt.show()


# num_labels decide how many the function will return
def get_top_incorrect_predictions(predictions, labels, num_labels=3):
    # Stores the counts of incorrect predictions for each label
    incorrect_counts = {}

    # Iterate over the predicted labels and true labels
    for prediction, label in zip(predictions, labels):
        # If the prediction is incorrect
        if prediction != label:
            # Increment the count of incorrect predictions for this label
            if label in incorrect_counts:
                incorrect_counts[label] += 1
            else:
                incorrect_counts[label] = 1

    # The num_labels with the highest number of incorrect predictions
    top_incorrect_labels = nlargest(num_labels, incorrect_counts, key=incorrect_counts.get)

    return top_incorrect_labels


def train_model(X, y):
    print("Training model...")
    # Convert the lists to numpy arrays
    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    # Prints count of labels and features
    len_y = len(y)
    print(f"labels length: {len_y}".format(len_y))
    len_x = len(X)
    print(f"features length: {len_x}".format(len_x))

    X = X.reshape(X.shape[0], -1)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    clf = SVC(kernel="linear")

    # Train the classifier on the training data
    clf.fit(X_train, y_train)

    print("\n shape")
    print(X_test.shape)
    print("\n")

    predictions = clf.predict(X_test)

    accuracy = evaluate_accuracy(predictions, y_test)

    print(f"Accuracy: {accuracy:.2f}")

    print("\n*Test*")
    print(y_test)
    print("--------------------------------------------------")
    print(predictions)
    print("*Prediction*\n")

    # Get the three labels with the highest number of incorrect predictions
    top_incorrect_labels = get_top_incorrect_predictions(predictions, y_test)

    # Print the top incorrect labels
    print(f"Top incorrect labels: {top_incorrect_labels}")

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

    return clf


def run():
    X, y = extract_data()
    model = train_model(X, y)

    with open("../models/svm_model.pkl", "wb") as f:
        pickle.dump(model, f)


run()
