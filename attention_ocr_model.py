import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import xml.etree.ElementTree as ET
from torch.nn import functional as F
from torchvision import transforms
import os
from PIL import Image


class OCRDataset(Dataset):
    def __init__(self, annotations_path, transforms=None):
        self.annotations = self.parse_annotations(annotations_path)
        self.transforms = transforms
    
    def parse_annotations(self, xml_path):
        annotations = []
        for filename in os.listdir(xml_path):
            if filename.endswith(".xml"):
                file_path = os.path.join(xml_path, filename)
                tree = ET.parse(file_path)
                root = tree.getroot()
                image_path = os.path.join(xml_path, root.find("filename").text)
                for obj in root.findall('object'):
                    label = obj.find('name').text
                    xmin = int(float(obj.find('bndbox/xmin').text))
                    ymin = int(float(obj.find('bndbox/ymin').text))
                    xmax = int(float(obj.find('bndbox/xmax').text))
                    ymax = int(float(obj.find('bndbox/ymax').text))
                    annotations.append({'label': label, 'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax, 'image_path': image_path})
        return annotations

    def normalize(self, tensor, mean, std):
        mean = torch.as_tensor(mean, dtype=torch.float32)
        std = torch.as_tensor(std, dtype=torch.float32)
        mean = mean.unsqueeze(1).unsqueeze(1)
        std = std.unsqueeze(1).unsqueeze(1)
        return tensor.sub_(mean).div_(std)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):

        annotation = self.annotations[idx]
        image_path = annotation['image_path']
        image = Image.open(image_path)
        label = annotation['label']

        if self.transforms:
            image = self.transforms(image)
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
            image = self.normalize(image, mean, std)
        
        return image, label


class AttentionOCR(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(AttentionOCR, self).__init__()
        self.conv = nn.Conv2d(1, 64, 3, padding=1)
        self.rnn = nn.GRU(input_size, hidden_size, num_layers=2, bidirectional=True)
        self.attn = nn.Linear(hidden_size * 2, 1)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def create_dataloader(self, annotations_path, batch_size, transforms, shuffle=True):
        
        dataset = OCRDataset(annotations_path, transforms)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        return dataloader

    def forward(self, x):
        # Pass the image through the convolutional layers
        x = self.conv(x)
        
        # Apply a RNN to the features
        x, _ = self.rnn(x)
        
        # Apply the attention mechanism
        attn_weights = self.attn(x)
        attn_weights = attn_weights.squeeze(2)
        attn_weights = F.softmax(attn_weights, dim=1)
        x = x * attn_weights.unsqueeze(2)
        
        # Pass the features through the fully connected layer
        x = self.fc(x)
        
        return x

    def train_attention_ocr(self, model, dataloader, criterion, optimizer, num_epochs):
        
        model.train()
        for epoch in range(num_epochs):
            for i, (inputs, labels) in enumerate(dataloader):
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                if (i+1) % 100 == 0:
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                           .format(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.item()))


# Instantiate the model
input_size = 256
hidden_size = 128
num_classes = 3000
model = AttentionOCR(input_size, hidden_size, num_classes)
print(model)

batch_size = 64
train_dataset = "./assets/cnn-annotations"

transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                transforms.RandomRotation(10),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataloader = model.create_dataloader(train_dataset, batch_size, transform)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
num_epochs = 20

model.train_attention_ocr(model, train_dataloader, criterion, optimizer, num_epochs)
