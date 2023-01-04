from PIL import Image
import pytesseract
import csv

# Simple image to string
print(pytesseract.image_to_string(Image.open('./assets/label-images/a-j-alexy.png')))

