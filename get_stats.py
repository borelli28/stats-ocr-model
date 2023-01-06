from PIL import Image
import os
import xml.etree.ElementTree as ET
import numpy as np


# import the model from ocr_model.py into this file


# OCR models read my labeled images
# Path to the images folder
image_folder = "./assets/labeled-images"

# Iterate over the images in the folder
for file in os.listdir(image_folder):
    # Ignore hidden files
    if file.startswith("."):
        continue
    
    # Construct the full path to the image file
    image_path = os.path.join(image_folder, file)
    
    # Load the image and extract the features
    image = Image.open(image_path)
    image_array = np.array(image, dtype=np.float32)
    image_array = image_array.reshape((1,) + image_array.shape)
    features = extract_features(image_array)

    features = features.reshape(1, -1)
    prediction = clf.predict(features)


    # Print the prediction
    print(f"Prediction for {file}: {prediction}")