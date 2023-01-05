from sklearn import svm
from PIL import Image
import json
import numpy as np


def load_data():
    # Open the images and get the pixel values as 2D arrays
    image_one = Image.open('./assets/labeled-images/a-j-alexy.png')
    pixels_one = np.array(image_one)
    image_two = Image.open('./assets/labeled-images/aaron-judge.png')
    pixels_two = np.array(image_two)

    # Stack the pixel arrays together
    X = np.vstack((pixels_one, pixels_two))

    # Load the annotations file
    annotations = None
    with open("./assets/annotations/2_coco_imglab.json", "r") as f:
        annotations = json.load(f)

    # Get the labels from the annotations
    labels = []
    for annotation in annotations["annotations"]:
        if "label" in annotation:
            labels.append(annotation["label"])

    # Convert the labels to a numpy array
    y = np.array(labels)

    return X, y

print(load_data())


# Load the training data
X_train, y_train = load_data()

# Make sure that X_train and y_train have the same number of samples
assert X_train.shape[0] == y_train.shape[0]

# Create the SVM model
svm_model = svm.SVC(kernel='linear')

# Train the model on the training data
svm_model.fit(X_train, y_train)

# Test the model on the test data
X_test, y_test = load_test_data()
accuracy = svm_model.score(X_test, y_test)

print(f"Test accuracy: {accuracy}")


