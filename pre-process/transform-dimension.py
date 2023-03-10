from PIL import Image

"""
Transform images dimensions into 2000x1650 by adding whitespace
"""

# Open image
img_path = "../assets/labeled-images/1.png"
im = Image.open(img_path)

# Get current size
width, height = im.size

# Create new image with the desired size
new_im = Image.new('RGB', (2000, 1650), (255, 255, 255))

# Paste the old image into the new image at the top left corner
new_im.paste(im, (0, 0))

# Convert to grayscale
new_im = new_im.convert('L')

# Save the new image
new_im.save(img_path)
