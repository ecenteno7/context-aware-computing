import os
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchaudio.transforms as T
import numpy as np
from sklearn.metrics import accuracy_score
import soundata


class AudioClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(AudioClassifier, self).__init__()
        # Define the convolutional layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)  # Conv1 layer
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # Conv2 layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # Max Pooling layer
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        
        # Dropout layer for regularization
        self.dropout = nn.Dropout(p=0.5)
        
        # Fully connected layers
        self.fc1 = None  # Will define dynamically based on the flattened size
        self.fc2 = nn.Linear(256, num_classes)  # Output layer (num_classes)

    def forward(self, x):
        # Apply convolution, batch normalization, and pooling in sequence
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))  # Conv1 + Pooling
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))  # Conv2 + Pooling
        
        # Calculate the flattened size dynamically
        flattened_size = x.size(1) * x.size(2) * x.size(3)
        
        # Define the first fully connected layer dynamically based on flattened size
        if self.fc1 is None:
            self.fc1 = nn.Linear(flattened_size, 256)
        
        # Flatten the tensor
        x = x.view(-1, flattened_size)  # Flatten the tensor for FC layers
        
        # Apply the fully connected layers with ReLU activation and dropout
        x = self.dropout(torch.relu(self.fc1(x)))  # Apply Dropout after FC1
        x = self.fc2(x)  # Output layer
        return x

