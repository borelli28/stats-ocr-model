import xml.etree.ElementTree as ET


# Path to the XML file
annotation_path = "./assets/annotations/jazz-annotations.xml"

def convert_to_box_file(annotation_path):
    # Load the XML file
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    # Open the box file for writing
    with open("./assets/annotations/jazz-annotations.box", "w") as f:
        # Iterate through all objects in the XML file
        for object in root.findall("object"):
            name = object.find("name").text
            bndbox = object.find("bndbox")
            xmin = bndbox.find("xmin").text
            ymin = bndbox.find("ymin").text
            xmax = bndbox.find("xmax").text
            ymax = bndbox.find("ymax").text

            # Write the data to the box file in the required format
            f.write(f"{name} {xmin} {ymin} {xmax} {ymax} 0\n")

# Call the function to convert the XML file to a box file
convert_to_box_file(annotation_path)
