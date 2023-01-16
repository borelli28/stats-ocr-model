import torch
import torch.nn as nn


class SimpleOCR(nn.Module):
    def __init__(self, num_classes):
        super(SimpleOCR, self).__init__()
        self.conv = nn.Conv2d(1, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 16 * 20, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(-1, 32 * 16 * 20)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
