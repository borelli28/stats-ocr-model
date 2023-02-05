from PIL import Image
import numpy as np
import joblib
import os
import xml.etree.ElementTree as ET


def extract_features(image):
    # Convert the image to a NumPy array with a consistent shape
    image_array = np.array(image, dtype=np.float32)
    image_array = image_array.reshape((1,) + image_array.shape)
    
    features = np.array(image_array).flatten()

    return features


def extract_data(image_path, annotations_path):
    annotation_folder = annotations_path
    X = []

    # Iterate over the annotations in the folder
    for file in os.listdir(annotation_folder):
        # Ignore hidden files
        if file.startswith("."):
            continue
        
        # Construct the full path to the annotation file
        file_path = os.path.join(annotation_folder, file)
        
        # Parse the XML tree from the file
        tree = ET.parse(file_path)

        # Find the filename of the image
        image_filename = tree.find("filename").text
        
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

    return X


def read_image(img_data):

    # Convert the lists to numpy arrays
    X = np.array(img_data, dtype=np.float32)

    len_x = len(X)
    print(f"img data length: {len_x}".format(len_x))

    X = X.reshape(X.shape[0], -1)

    prediction = svm_model.predict(X)

    return prediction


svm_model = joblib.load("./models/svm_model.pkl")
image_path = "./assets/labeled-images/aaron-judge.png"
img_data = extract_data()
read_image(img_data)




