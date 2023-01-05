import pytesseract
from PIL import Image
import json
import sklearn
from sklearn import model_selection, linear_model
import numpy as np


# Open an image and get its pixel values as a 2D array
image_one = Image.open('./assets/labeled-images/a-j-alexy.png')
pixels_one = np.array(image_one)

image_two = Image.open('./assets/labeled-images/aaron-judge.png')
pixels_two = np.array(image_two)


# Load the annotations file
annotations = None
with open("./assets/annotations/2_coco_imglab.json", "r") as f:
    annotations = json.load(f)
    # print(annotations)

# Create a mapping from category id to category name
categories_dict = {}
for category in annotations["categories"]:
    categories_dict[category["id"]] = category["name"]

# Extract the labels from the annotations
labels = []
for annotation in annotations["annotations"]:
    category_id = annotation["category_id"]
    if category_id in categories_dict:
        label = categories_dict[category_id]
        labels.append(label)


# Stack the pixel arrays together
X = np.vstack((pixels_one, pixels_two))
# Reshape the array into a 2D array of shape (n_samples, n_features)
X = X.reshape(-1, pixels_one.size)
y = labels

print("X & y size:")
print(len(X))
print(len(labels))

# Split the data into training and test sets
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=1, random_state=42)

# Choose a classification algorithm and instantiate a model object
model = sklearn.linear_model.LogisticRegression(max_iter=1000)

# Train the model on the training data
model.fit(X_train, y_train)

# Evaluate the model on the test data
score = model.score(X_test, y_test)
print(f"Model score: {score}".format(score))




