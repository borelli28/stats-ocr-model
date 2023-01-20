import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import xml.etree.ElementTree as ET
import os
from PIL import Image
from torchvision import transforms


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])


class CustomDataset(Dataset):
    def __init__(self, annotations_path, images_path, classes=None):
        self.classes = classes
        self.images_path = images_path
        self.annotations_path = annotations_path
        self.annotations = self.parse_annotations()

    # Returns annotations in a format that the model prefers
    def parse_annotations(self):
        print("parse_annotations()")
        annotations = []
        classes = set()

        for filename in os.listdir(self.annotations_path):
            if filename.endswith(".xml"):
                file_path = os.path.join(self.annotations_path, filename)
                tree = ET.parse(file_path)
                root = tree.getroot()
                image_name = root.find("filename").text
                image_path = os.path.join(self.images_path, image_name)
                for obj in root.findall("object"):
                    label = obj.find("name").text
                    classes.add(label)
                    xmin = int(float(obj.find("bndbox/xmin").text))
                    ymin = int(float(obj.find("bndbox/ymin").text))
                    xmax = int(float(obj.find("bndbox/xmax").text))
                    ymax = int(float(obj.find("bndbox/ymax").text))
                    annotations.append({
                        "label": label, "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax, "image_path": image_path})
        
        self.classes = list(classes)

        return annotations

    def __len__(self):
        print("__len__()")
        return len(self.annotations)

    def __getitem__(self, idx):
        print("__getitem__()")
        annotation = self.annotations[idx]
        image_path = annotation["image_path"]
        image = Image.open(os.path.join(image_path))
        image = transform(image)
        label = self.classes.index(annotation["label"])

        return image, label


class SimpleOCR(nn.Module):
    def __init__(self, num_classes, batch_size):
        super(SimpleOCR, self).__init__()

        self.batch_size = batch_size
        
        # Convolutional layer to extract features from input image
        self.conv = nn.Conv2d(1, 32, 3, padding=1)
        
        # Max pooling layer to reduce spatial resolution of feature maps
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layer to classify characters in the image
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    # Defines how the input data is processed and passed through the layers
    def forward(self, x):
        print("forward()")
        # Pass input through convolutional layers
        x = self.conv(x)
        x = self.pool(x)

        print("self.pool(x)")
        print(x.shape)
        print("\nexit: self.pool(x)\n")

        # Reshape feature maps for fully connected layers
        x = x.view(x.size(0), -1)

        print("x.view():")
        print(x.shape)
        
        # Pass features through fully connected layers
        x = self.fc1(x)
        x = self.fc2(x)

        print("x = self.fc2(x)")
        print(x.shape)
        print("\n")

        return x


def train(annotations_path, images_path, batch_size, num_epochs):
    # Initialize CustomDataset class in order to extract the num_classes
    dataset = CustomDataset(annotations_path, images_path)
    num_classes = len(dataset.classes)

    # Initialize the model
    model = SimpleOCR(num_classes, batch_size)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Create a dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print("DataLoader:")
    print(dataloader)

    # Train the model
    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(dataloader):
            print("\ninputs:")
            print(inputs.shape)
            print("labels:")
            print(labels.shape)
            print("\n")
            
            optimizer.zero_grad()
            outputs = model(inputs)
            print("\ninputs:")
            print(inputs.shape)
            print("labels:")
            print(labels.shape)
            print("\n")
            print("\noutputs:")
            print(outputs.shape)
            print("\n")
            
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
