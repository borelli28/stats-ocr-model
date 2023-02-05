from PIL import Image
import numpy as np
import joblib


# Load the SVM model from the .pkl file
svm_model = joblib.load("./models/svm_model.pkl")


def extract_features(image):
    # Convert the image to a NumPy array with a consistent shape
    image_array = np.array(image, dtype=np.float32)
    image_array = image_array.reshape((1,) + image_array.shape)
    
    features = np.array(image_array).flatten()

    return features


def read_image(image_path):
    image = Image.open(image_path)

    # Convert the image to a NumPy array with a consistent shape
    image_array = np.array(image, dtype=np.float32)
    image_array = image_array.reshape((1,) + image_array.shape)

    # Extract features from the image
    features = extract_features(image_array)

    print("\n shape")
    print(features.shape)
    print("\n")

    features = features.reshape(features.shape[0], -1)

    print("\n shape")
    print(features.shape)
    print("\n")

    # Make a prediction using the model
    prediction = svm_model.predict(features)
    print("\npredictioooon:")
    print(prediction)
    print(prediction.size)
    print(type(prediction))
    print("\n")

    for i in range(len(prediction)):
        print(prediction[i])

    return prediction


image_path = "./assets/labeled-images/aaron-judge.png"
read_image(image_path)




