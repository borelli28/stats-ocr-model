from PIL import Image, ImageEnhance
import pytesseract
import csv

# Image to string
# print(pytesseract.image_to_string(Image.open("./assets/label-images/a-j-alexy.png")))

# Load image
image = Image.open("./assets/label-images/a-j-alexy.png")


# Examine file contents
print(image.format, image.size, image.mode)


# Convert the image to grayscale
image = image.convert('L')

# Save the grayscale image
image.save('image_grayscale.jpg')


# Create an ImageEnhancer object
enhancer = ImageEnhance.Contrast(image)

# Adjust the contrast
image = enhancer.enhance(1.5)

# Save the enhanced image
image.save('image_enhanced.jpg')




# print(pytesseract.image_to_string(Image.open("image_enhanced.jpg")))

