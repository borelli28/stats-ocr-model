import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import xml.etree.ElementTree as ET
import os
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, annotations_path, images_path, transforms=None):
        self.annotations = self.parse_annotations(annotations_path)
        self.images_path = images_path
        self.transforms = transforms
    
    def parse_annotations(self, xml_path, images_path):
        annotations = []
        for filename in os.listdir(xml_path):
            if filename.endswith(".xml"):
                file_path = os.path.join(xml_path, filename)
                tree = ET.parse(file_path)
                root = tree.getroot()
                image_name = root.find("filename").text
                image_path = os.path.join(images_path, image_name)
                for obj in root.findall("object"):
                    label = obj.find("name").text
                    xmin = int(float(obj.find("bndbox/xmin").text))
                    ymin = int(float(obj.find("bndbox/ymin").text))
                    xmax = int(float(obj.find("bndbox/xmax").text))
                    ymax = int(float(obj.find("bndbox/ymax").text))
                    annotations.append({
                        "label": label, "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax, "image_path": image_path})
        return annotations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        image_path = annotation["image_path"]
        image = Image.open(os.path.join(self.images_path, image_path))
        label = annotation["label"]
        
        return image, label


class SimpleOCR(nn.Module):
    def __init__(self, num_classes, batch_size):
        super(SimpleOCR, self).__init__()
        
        # Convolutional layer to extract features from input image
        self.conv = nn.Conv2d(1, 32, 3, padding=1)
        
        # Max pooling layer to reduce spatial resolution of feature maps
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layer to classify characters in the image
        self.fc1 = nn.Linear(32 * 16 * 20, 128) 
        self.fc2 = nn.Linear(128, num_classes)

        self.batch_size = batch_size

    # Defines how the input data is processed and passed through the layers
    def forward(self, x):
        # Pass input through convolutional layers
        x = self.conv(x)
        x = self.pool(x)
        
        # Reshape feature maps for fully connected layers
        x = x.view(-1, 32 * 16 * 20)
        
        # Pass features through fully connected layers
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def train(annotations_path, images_path, batch_size, num_epochs):
    # Initialize the model
    model = SimpleOCR(annotations_path, images_path)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Create a dataset from the annotations
    dataset = CustomDataset(annotations_path)

    # Create a dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Train the model
    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Checks if the current step (i+1) is divisible by 10
            # Every 10 steps the progress is printed
            if (i+1) % 10 == 0:
                print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}" 
                       .format(
                        epoch+1, num_epochs, i+1, len(dataloader), loss.item()))


annotations_path = "./assets/annotations"
images_path = "./assets/labeled-images"
train(annotations_path, images_path, 64, 10)



#
