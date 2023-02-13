import cv2
import numpy as np
import xml.etree.ElementTree as ET
import easyocr

# Set up EasyOCR
reader = easyocr.Reader(['en'])

img_path = "./assets/labeled-images/1.png"
img = cv2.imread(img_path)

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur to reduce noise
gray = cv2.GaussianBlur(gray, (5, 5), 0)

# Threshold the image to convert it to binary
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# Perform vertical projection to find the location of each line of numbers
vertical_projection = np.sum(thresh, axis=1)
y_coords = np.where(vertical_projection > 0)[0]

# Create the root element for the XML file
root = ET.Element("annotation")
ET.SubElement(root, "folder").text = "images"
ET.SubElement(root, "filename").text = "mike-brosseau.png"
ET.SubElement(root, "path").text = img_path
source = ET.SubElement(root, "source")
ET.SubElement(source, "database").text = "Unknown"
size = ET.SubElement(root, "size")
height, width, depth = img.shape
ET.SubElement(size, "width").text = str(width)
ET.SubElement(size, "height").text = str(height)
ET.SubElement(size, "depth").text = str(depth)
ET.SubElement(root, "segmented").text = "0"

# Loop through each line of numbers
for i in range(len(y_coords) - 1):
    y1 = y_coords[i]
    y2 = y_coords[i + 1]

    line_img = img[y1:y2, :]
    line_gray = gray[y1:y2, :]
    line_thresh = thresh[y1:y2, :]

    # Use EasyOCR to find the digits in the line
    results = reader.readtext(line_img)

    # Loop through each result and add it as an object to the XML tree
    for result in results:
        # Get the bounding box for the result
        bbox = result[0]
        x, y, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]

        # Create the XML element for the object
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = "digit"
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        bbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bbox, "xmin").text = str(x)
        ET.SubElement(bbox, "ymin").text = str(y)
        ET.SubElement(bbox, "xmax").text = str(x + w)
        ET.SubElement(bbox, "ymax").text = str(y + h)

        # Draw the bounding box on the image
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

# Save the annotated image
cv2.imwrite("annotated_image.png", img)

# Write the XML file
tree = ET.ElementTree(root)
tree.write("numbers.xml", encoding="utf-8", xml_declaration=True)
