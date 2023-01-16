import torch
import torch.nn as nn

class SimpleOCR(nn.Module):
    def __init__(self, num_classes):
        super(SimpleOCR, self).__init__()
        
        # Convolutional layer to extract features from input image
        self.conv = nn.Conv2d(1, 32, 3, padding=1)
        
        # Max pooling layer to reduce spatial resolution of feature maps
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layer to classify characters in the image
        self.fc1 = nn.Linear(32 * 16 * 20, 128) 
        self.fc2 = nn.Linear(128, num_classes)

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
