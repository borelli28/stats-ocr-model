from PIL import Image, ImageEnhance, ImageFilter
import pytesseract


# Load image
image = Image.open("./assets/label-images/aaron-judge.png")


# Convert the image to grayscale
image = image.convert('L')
# image.save('image_grayscale.jpg')

# Crop the image
# 						(left, upper, right, lower)
cropped_image = image.crop((0, 350, 2000, 2000))
cropped_image.save('cropped_image.jpg')

# Image to string
print(pytesseract.image_to_string(Image.open("cropped_image.jpg")))


# # Create an ImageEnhancer object
# enhancer = ImageEnhance.Contrast(cropped_image)
# # Adjust the contrast
# cropped_image = enhancer.enhance(5)
# cropped_image.save('image_enhanced.jpg')

# print(pytesseract.image_to_string(Image.open("image_enhanced.jpg")))

