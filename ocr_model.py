from PIL import Image
import json
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = '/path/to/tesseract'

# Set the path to the trained model
tessdata_dir_config = '--tessdata-dir "/path/to/trained/model"'

# Read the image and recognize the text
text = pytesseract.image_to_string(image, lang='eng', config=tessdata_dir_config)