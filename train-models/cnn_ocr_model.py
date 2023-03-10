import torch
from torch import nn
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import xml.etree.ElementTree as ET
import os
from PIL import Image
from tqdm import tqdm


class CustomDataset(Dataset):
    def __init__(self, annotations_path, images_path, labels=None):
        self.labels = labels
        self.images_path = images_path
        self.annotations_path = annotations_path
        self.annotations = self.parse_annotations()

    # Returns annotations in a format that the model prefers
    def parse_annotations(self):
        annotations = []
        labels = set()

        for filename in os.listdir(self.annotations_path):
            if filename.endswith(".xml"):
                file_path = os.path.join(self.annotations_path, filename)
                tree = ET.parse(file_path)
                root = tree.getroot()
                image_name = root.find("filename").text
                image_path = os.path.join(self.images_path, image_name)
                for obj in root.findall("object"):
                    label = obj.find("name").text
                    labels.add(label)
                    xmin = int(float(obj.find("bndbox/xmin").text))
                    ymin = int(float(obj.find("bndbox/ymin").text))
                    xmax = int(float(obj.find("bndbox/xmax").text))
                    ymax = int(float(obj.find("bndbox/ymax").text))
                    annotations.append({
                        "label": label, "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax, "image_path": image_path})
        
        self.labels = list(labels)
        print("labels in annotations: {len_labels}".format(len_labels=str(len(labels))))

        return annotations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        image_path = annotation["image_path"]
        image = Image.open(os.path.join(image_path))
        image = transform(image)
        label = self.labels.index(annotation["label"])

        return image, label


# Define model
class CNN_OCR(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # The first convolutional layer takes in a 1-channel image and applies 6 filters of kernel size 5
        self.conv1 = nn.Conv2d(1, 6, 5)
        # Max pooling layer with kernel size 2x2
        self.pool = nn.MaxPool2d(2, 2)
        # The second convolutional layer takes in the output of the first convolutional layer and applies 16 filters of kernel size 5
        self.conv2 = nn.Conv2d(6, 16, 5)
        # Fully connected layers with 120, 84 and num_classes neurons respectively
        self.fc1 = nn.Linear(16 * 409 * 497, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        print(f"Num of classes: {num_classes}")

    def forward(self, x):
        # Apply first convolutional layer, ReLU activation function, and max pooling
        x = self.pool(F.relu(self.conv1(x)))
        # Apply second convolutional layer, ReLU activation function, and max pooling
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten the output of the second convolutional layer
        x = torch.flatten(x, 1)
        # Apply fully connected layers with ReLU activation function
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Apply final fully connected layer
        x = self.fc3(x)

        return x


def train(annotations_path, images_path, batch_size, num_epochs, num_classes):
    try:
        dataset = CustomDataset(annotations_path, images_path)

        model = CNN_OCR(num_classes)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size

        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

        with tqdm(total=num_epochs, desc="Epochs") as pbar_epoch:
            for epoch in range(num_epochs):

                with tqdm(total=len(train_dataloader)) as pbar_batch:
                    for i, (inputs, labels) in enumerate(train_dataloader):
                        pbar_batch.set_description(
                            f"Training batch #{i+1}".format(i+1))

                        optimizer.zero_grad()
                        outputs = model(inputs)
                        
                        loss = loss_fn(outputs, labels)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        pbar_batch.update(1)

                pbar_epoch.update(1)

            test(model, test_dataloader, loss_fn)

            return model
    except Exception as error:
        print(error)


# Evaluate the model accuracy based on test data
def test(model, test_dataloader, loss_function):
    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)
    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in test_dataloader:
            pred = model(X)
            test_loss += loss_function(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Model performance test data: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


transform = transforms.Compose([
    transforms.ToTensor()
])

batch_size = 10
num_epochs = 10


classes = set()
for filename in os.listdir('../assets/annotations'):
    if filename.endswith('.xml'):
        tree = ET.parse(os.path.join('../assets/annotations', filename))
        root = tree.getroot()
        for obj in root.findall('object'):
            obj_name = obj.find('name').text
            classes.add(obj_name)

num_classes = len(classes)
print(f'Number of classes: {num_classes}')
# num_classes = 83


annotations_path = "../assets/annotations"
images_path = "../assets/labeled-images"
model = train(annotations_path, images_path, batch_size, num_epochs, num_classes)

torch.save(model.state_dict(), "../models/cnn_model.pth")
print("Saved PyTorch Model State to ./models/cnn_model.pth")
