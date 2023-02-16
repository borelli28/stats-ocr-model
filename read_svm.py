import cv2
import joblib
import numpy as np
import os
import xml.etree.ElementTree as ET
from PIL import Image


def create_annotation(image_path, object_name, xmin, ymin, xmax, ymax, width, height):
    # create annotation element
    annotation = ET.Element("annotation")

    # create sub elements
    folder = ET.SubElement(annotation, "folder")
    folder.text = os.path.dirname(image_path)

    filename = ET.SubElement(annotation, "filename")
    filename.text = os.path.basename(image_path)

    path = ET.SubElement(annotation, "path")
    path.text = image_path

    source = ET.SubElement(annotation, "source")
    database = ET.SubElement(source, "database")
    database.text = "Unknown"

    size = ET.SubElement(annotation, "size")
    w = ET.SubElement(size, "width")
    w.text = str(width)
    h = ET.SubElement(size, "height")
    h.text = str(height)
    d = ET.SubElement(size, "depth")
    d.text = "3"

    segmented = ET.SubElement(annotation, "segmented")
    segmented.text = "0"

    # create object element
    obj = ET.SubElement(annotation, "object")
    name = ET.SubElement(obj, "name")
    name.text = object_name
    pose = ET.SubElement(obj, "pose")
    pose.text = "Unspecified"
    truncated = ET.SubElement(obj, "truncated")
    truncated.text = "0"
    difficult = ET.SubElement(obj, "difficult")
    difficult.text = "0"
    bndbox = ET.SubElement(obj, "bndbox")
    xmin_element = ET.SubElement(bndbox, "xmin")
    xmin_element.text = str(xmin)
    ymin_element = ET.SubElement(bndbox, "ymin")
    ymin_element.text = str(ymin)
    xmax_element = ET.SubElement(bndbox, "xmax")
    xmax_element.text = str(xmax)
    ymax_element = ET.SubElement(bndbox, "ymax")
    ymax_element.text = str(ymax)

    # create an ElementTree object
    tree = ET.ElementTree(annotation)

    # write the annotation to file
    annotation_dir = "./read-annotations"
    annotation_path = os.path.join(annotation_dir, os.path.splitext(os.path.basename(image_path))[0] + ".xml")
    tree.write(annotation_path)

    return annotation_path


def extract_features(image):
    # Convert the image to a NumPy array with a consistent shape
    image_array = np.array(image, dtype=np.float32)
    image_array = image_array.reshape((1,) + image_array.shape)
    
    features = np.array(image_array).flatten()

    return features


def extract_labels(annotation):
    labels = []
    
    # Iterate over the objects in the annotation
    for obj in annotation.findall("object"):
        # Extract the label (i.e., the text in the image)
        label = obj.find("name").text
        labels.append(label)

    return labels


def extract_data():
    print("Extracting data from training data...")
    annotation_folder = "./read-annotations/"

    # List to store the features and labels
    X = []
    y = []

    # Iterate over the annotations in the folder
    for file in os.listdir(annotation_folder):
        # Ignore hidden files
        if file.startswith("."):
            continue
        
        # Construct the full path to the annotation file
        file_path = os.path.join(annotation_folder, file)
        
        # Parse the XML tree from the file
        tree = ET.parse(file_path)

        # Extract the labels from the annotation file
        labels = extract_labels(tree)

        # Add labels to y
        y.extend(labels)

        # Find the filename of the image
        image_filename = tree.find("filename").text
        # if the filename does not have an extension, add one
        if image_filename[-1] != "g":
            image_filename = image_filename + ".png"
        
        # Construct the full path to the image file
        image_path = os.path.join("./assets/labeled-images", image_filename)
        
        # Open the image
        image = Image.open(image_path)
        
        # Iterate over the objects in the annotation
        for obj in tree.findall("object"):
            
            # Extract the bounding box coordinates
            xmin = int(round(float(obj.find("bndbox/xmin").text)))
            ymin = int(round(float(obj.find("bndbox/ymin").text)))
            xmax = int(round(float(obj.find("bndbox/xmax").text)))
            ymax = int(round(float(obj.find("bndbox/ymax").text)))
            
            # Crop the image to the bounding box
            cropped_image = image.crop((xmin, ymin, xmax, ymax))
            
            # Resize the image to a fixed size
            resized_image = cropped_image.resize((200, 200))

            # Convert the image to a NumPy array with a consistent shape
            image_array = np.array(resized_image, dtype=np.float32)
            image_array = image_array.reshape((1,) + image_array.shape)

            features = extract_features(image_array)

            # Add the features to the list
            X.append(features)

    return X, y


def draw_image(image_path):
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Otsu's thresholding
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours in the binary image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Extract the regions of the image containing text
    text_regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        text_regions.append(image[y:y+h, x:x+w])

    # Process each text region with your OCR model
    for i, region in enumerate(text_regions):
        # Use your SVM OCR trained model to predict the labels for this region
        # labels = svm_model.predict(region)

        # Draw a rectangle around the text region
        contour = contours[i]
        if len(contour) >= 4:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Create annotation file for this object
            object_name = "text_region"  # replace with appropriate object name
            annotation = create_annotation(image_path, object_name, x, y, x+w, y+h, image.shape[1], image.shape[0])
            print(annotation)

            # Add the predicted labels as text within the rectangle
            # label_text = ''.join(labels)
            # cv2.putText(image, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the image with text regions and labels
    cv2.imshow("Text Regions with Labels", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


svm_model = joblib.load("./models/svm_model.pkl")
# Load the image
image_path = "./assets/labeled-images/jazz-chisholm-jr.png"

draw_image(image_path)

