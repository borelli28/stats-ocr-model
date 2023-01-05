from PIL import Image
import numpy as np
import pytesseract
import os


# Path to the images folder
image_folder = "./assets/labeled-images"

# Iterate over the images in the folder
for file in os.listdir(image_folder):
    # Ignore hidden files
    if file.startswith("."):
        continue
    
    # Construct the full path to the image file
    file_path = os.path.join(image_folder, file)
    
    # Open the image
    image = Image.open(file_path)
    
    # Read the text in the image
    text = pytesseract.image_to_string(image)
    
    # Print the text
    print(text)


# pytesseract.pytesseract.tesseract_cmd = '/path/to/tesseract'

# # Set the path to the trained model
# tessdata_dir_config = '--tessdata-dir "/path/to/trained/model"'

# # Read the image and recognize the text
# text = pytesseract.image_to_string(image, lang='eng', config=tessdata_dir_config)