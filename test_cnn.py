import torch
from torch import nn
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import xml.etree.ElementTree as ET
import os
from PIL import Image
from torchvision import transforms


transform = transforms.Compose([
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
        # print("__getitem__()")
        annotation = self.annotations[idx]
        image_path = annotation["image_path"]
        image = Image.open(os.path.join(image_path))
        image = transform(image)
        label = self.classes.index(annotation["label"])

        return image, label

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        # logits = self.linear_relu_stack(x)
        return x


model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


def train(annotations_path, images_path, batch_size, num_epochs):
    # Initialize CustomDataset class in order to extract the num_classes
    dataset = CustomDataset(annotations_path, images_path)
    num_classes = len(dataset.classes)

    # Initialize the model
    model = NeuralNetwork()

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Create a dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create data loaders
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)


    # Train the model
    for epoch in range(num_epochs):
        print(f"epoch: {epoch}".format(epoch))
        counter = 0
        print(len(train_dataloader))
        for i, (inputs, labels) in enumerate(train_dataloader):
            print(print(f"counter: {counter}".format(counter)))
            
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
            # loss.backward()
            optimizer.step()

            # Every 10 steps the progress is printed
            if (i+1) % 10 == 0:
                print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}" 
                       .format(
                        epoch+1, num_epochs, i+1, len(dataloader), loss.item()))

            counter += 1

    # Test
    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return model


batch_size = 64
num_epochs = 10
annotations_path = "./assets/annotations"
images_path = "./assets/labeled-images"
model = train(annotations_path, images_path, batch_size, num_epochs)

# Save model
torch.save(model.state_dict(), "./models/cnn_model.pth")
print("Saved PyTorch Model State to ./models/cnn_model.pth")

# for t in range(num_epochs):
#     print(f"Epoch {t+1}\n-------------------------------")
#     train(train_dataloader, model, loss_fn, optimizer)
#     # test(test_dataloader, model, loss_fn)
# print("Done!")










#