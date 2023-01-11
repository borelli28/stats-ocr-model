import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim


class AttentionOCR(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(AttentionOCR, self).__init__()
        self.conv = nn.Conv2d(1, 64, 3, padding=1)
        self.rnn = nn.GRU(input_size, hidden_size, num_layers=2, bidirectional=True)
        self.attn = nn.Linear(hidden_size * 2, 1)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

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

    def create_dataloader(dataset, batch_size, shuffle=True):
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


    def train_attention_ocr(model, dataloader, criterion, optimizer, num_epochs):
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
train_dataset = "./assets/"

train_dataloader = model.create_dataloader(train_dataset, batch_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
num_epochs = 20

model.train_attention_ocr(model, train_dataloader, criterion, optimizer, num_epochs)
