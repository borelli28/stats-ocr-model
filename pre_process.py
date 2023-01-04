from PIL import Image, ImageEnhance, ImageFilter
import csv


# Open the file for reading
with open('players.csv', 'r') as csvfile:
    # Create a CSV reader object
    reader = csv.reader(csvfile)

    # Skip the first row (column names)
    next(reader)

    # Read the rows of the file
    for row in reader:
        # Get player name from players.csv
        name = str(row[0])

        print(f"Working with {name} image...")

		# Load image
        image = Image.open(f"./assets/label-images/{name}.png".format(name))

		# Convert the image to grayscale
        image = image.convert('L')
		# image.save('image_grayscale.png')

		# Crop the image
		# 						(left, upper, right, lower)
        cropped_image = image.crop((0, 350, 2000, 2000))
        cropped_image.save(f"./assets/label-images/{name}.png".format(name))

        print(f"Done with {name} image")





# # Increase contrast of image
# # Create an ImageEnhancer object
# enhancer = ImageEnhance.Contrast(cropped_image)
# # Adjust the contrast
# cropped_image = enhancer.enhance(5)
# cropped_image.save('image_enhanced.jpg')

# print(pytesseract.image_to_string(Image.open("image_enhanced.jpg")))

