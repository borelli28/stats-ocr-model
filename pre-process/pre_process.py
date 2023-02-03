from PIL import Image
import csv


# Open the file for reading
with open('../get-images/players.csv', 'r') as csvfile:
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
        image = Image.open(f"../assets/label-images/{name}.png".format(name))

		# Convert the image to grayscale
        image = image.convert('L')
		# image.save('image_grayscale.png')

		# Crop the image
		# 						(left, upper, right, lower)
        cropped_image = image.crop((0, 350, 2000, 2000))
        cropped_image.save(f"../assets/label-images/{name}.png".format(name))

        print(f"Done with {name} image")
