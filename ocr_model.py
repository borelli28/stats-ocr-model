from PIL import Image
import json
import numpy as np
import pytesseract


# Open the image
image = Image.open("./assets/labeled-images/a-j-alexy.png")


# Reads the text in the image
text = pytesseract.image_to_string(image)
print(text)


# pytesseract.pytesseract.tesseract_cmd = '/path/to/tesseract'

# # Set the path to the trained model
# tessdata_dir_config = '--tessdata-dir "/path/to/trained/model"'

# # Read the image and recognize the text
# text = pytesseract.image_to_string(image, lang='eng', config=tessdata_dir_config)