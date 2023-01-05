import xml.etree.ElementTree as ET


# Load the XML file
tree = ET.parse("./assets/annotations/jazz-annotations.xml")
root = tree.getroot()

# Open the output file
with open("./assets/annotations/jazz-annotations.box", "w") as f:
    # Iterate over all objects in the XML file
    for obj in root.iter("object"):
        # Get the name of the object and write it to the output file
        name = obj.find("name").text
        f.write(name + " ")
        
        # Get the bounding box coordinates and write them to the output file
        bndbox = obj.find('bndbox')
        xmin = bndbox.find('xmin').text
        ymin = bndbox.find('ymin').text
        xmax = bndbox.find('xmax').text
        ymax = bndbox.find('ymax').text
        f.write(xmin + ' ' + ymin + ' ' + xmax + ' ' + ymax + ' ')
        
        # Write the page number (assume it is always 0 in this case)
        f.write("0\n")