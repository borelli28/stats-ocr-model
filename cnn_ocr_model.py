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
    def __init__(self, annotations_path, images_path, classes=None):
        self.classes = classes
        self.images_path = images_path
        self.annotations_path = annotations_path
        self.annotations = self.parse_annotations()

    # Returns annotations in a format that the model prefers
    def parse_annotations(self):
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
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        image_path = annotation["image_path"]
        image = Image.open(os.path.join(image_path))
        image = transform(image)
        label = self.classes.index(annotation["label"])

        return image, label


# Define model
class CNN_OCR(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 409 * 497, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        # print(f"Num of classes: {num_classes}")

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


def train(annotations_path, images_path, batch_size, num_epochs):
    dataset = CustomDataset(annotations_path, images_path)
    num_classes = len(dataset.classes)

    model = CNN_OCR(num_classes)

    # Define the loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    # Create data loaders
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    with tqdm(total=num_epochs, desc="Epoch") as pbar_epoch:
        for epoch in range(num_epochs):

            with tqdm(total=len(train_dataloader), desc="Batch") as pbar_batch:
                for i, (inputs, labels) in enumerate(train_dataloader):
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    
                    loss = loss_fn(outputs, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    pbar_batch.update(1)

            pbar_epoch.update(1)

        test(test_dataloader, loss_fn)

        return model


def test(test_dataloader, loss_function):
    # Evaluate the model accuracy based on test data
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

batch_size = 256
num_epochs = 1
annotations_path = "./assets/annotations"
images_path = "./assets/labeled-images"
model = train(annotations_path, images_path, batch_size, num_epochs)

# Save model
torch.save(model.state_dict(), "./models/cnn_model.pth")
print("Saved PyTorch Model State to ./models/cnn_model.pth")
