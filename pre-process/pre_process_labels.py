import os
import xml.etree.ElementTree as ET


# Define a function to extract labels from the annotations
def extract_labels(annotation):
    labels = []
    
    # Iterate over the objects in the annotation
    for obj in annotation.findall("object"):
        # Extract the label (i.e., the text in the image)
        label = obj.find("name").text
        labels.append(label)

    return labels


def load_labels(annotations_path):
    labels = []
    for file in os.listdir(annotations_path):
        if not file.startswith("."):
            file_path = os.path.join(annotations_path, file)
            tree = ET.parse(file_path)
            label = extract_labels(tree)
            labels.extend(label)
    return labels


MAX_LABEL_LENGTH = 4

ACCEPTABLE_WORDS = ["teams", "Season", "GO/AO", "Draft", "Career", "Stats",
"Regular", "BABIP", "Advanced", "GIDPO", "MLB", "2 teams", "Position"]


# Find all the words that we want to cut from the annotations
def preprocess_labels(labels):
    new_labels = []
    for label in labels:
        if len(label) > MAX_LABEL_LENGTH and label not in ACCEPTABLE_WORDS:
            new_labels.append(label)
    return new_labels


def remove_unwanted_words(annotations_path, unwanted_words):
    for file in os.listdir(annotations_path):
        if not file.startswith("."):
            file_path = os.path.join(annotations_path, file)
            tree = ET.parse(file_path)
            root = tree.getroot()
            # Iterate over the objects in the annotation
            for obj in root.findall("object"):
                label = obj.find("name").text
                # Check if the label is in the list of unwanted words
                if label in unwanted_words:
                    root.remove(obj)
            # Write the updated annotation file
            tree.write(file_path)


annotations_path = "../assets/annotations"

labels = load_labels(annotations_path)

unwanted_words = preprocess_labels(labels)
print(unwanted_words)


remove_unwanted_words(annotations_path, unwanted_words)
print("Words removed from annotations")
