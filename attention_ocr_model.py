import torch
import torch.nn as nn


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


# Instantiate the model
input_size = 256
hidden_size = 128
num_classes = 3000
model = AttentionOCR(input_size, hidden_size, num_classes)
print(model)
