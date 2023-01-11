import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import xml.etree.ElementTree as ET
from torch.nn import functional as F
from torchvision import transforms
import os


class AttentionOCR(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(AttentionOCR, self).__init__()
        self.conv = nn.Conv2d(1, 64, 3, padding=1)
        self.rnn = nn.GRU(input_size, hidden_size, num_layers=2, bidirectional=True)
        self.attn = nn.Linear(hidden_size * 2, 1)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def parse_annotations(self, xml_path):

        labels = []
        for filename in os.listdir(xml_path):
            if filename.endswith(".xml"):
                file_path = os.path.join(xml_path, filename)
                tree = ET.parse(file_path)
                root = tree.getroot()
                for obj in root.findall('object'):
                    label = obj.find('name').text
                    xmin = int(float(obj.find('bndbox/xmin').text))
                    ymin = int(float(obj.find('bndbox/ymin').text))
                    xmax = int(float(obj.find('bndbox/xmax').text))
                    ymax = int(float(obj.find('bndbox/ymax').text))
                    labels.append({'label': label, 'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax})
        return labels

    def create_dataloader(self, annotations_path, batch_size, transforms, shuffle=True):
        
        annotations = self.parse_annotations(annotations_path)
        dataset = torch.utils.data.Dataset(annotations, transforms)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        return dataloader

    def forward(self, x):
        
        x = self.conv(x)
        x, _ = self.rnn(x)
        attn_weights = self.attn(x)
        attn_weights = attn_weights.squeeze(2)
        attn_weights = F.softmax(attn_weights, dim=1)
        x = x * attn_weights.unsqueeze(2)
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
train_dataset = "./assets/annotations"

transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                transforms.RandomRotation(10),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataloader = model.create_dataloader(train_dataset, batch_size, transform)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
num_epochs = 20

model.train_attention_ocr(model, train_dataloader, criterion, optimizer, num_epochs)
