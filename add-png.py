import os
import glob
import xml.etree.ElementTree as ET

annotations_path = "./assets/annotations"

for xml_file in glob.glob(os.path.join(annotations_path, "*.xml")):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    filename = root.find("filename").text
    
    if not filename.endswith(".png"):
        filename += ".png"
        root.find("filename").text = filename
    
    tree.write(xml_file)
