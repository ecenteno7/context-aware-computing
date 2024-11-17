import os
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
import torchaudio.transforms as T
import numpy as np
from sklearn.metrics import accuracy_score
import soundata

# Hyperparameters
num_epochs = 1
batch_size = 16  # Lowered batch size for quicker training and memory efficiency
learning_rate = 0.001
num_classes = 10  # Adjust based on your dataset
max_length = 1500  # Max length for audio processing
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET = 'urbansound8k'

# Function to pad or truncate the Mel spectrogram to a fixed length
def pad_or_truncate(mel_spec, max_length):
    length = mel_spec.size(2)
    if length < max_length:
        pad = max_length - length
        mel_spec = torch.nn.functional.pad(mel_spec, (0, pad))
    elif length > max_length:
        mel_spec = mel_spec[:, :, :max_length]
    return mel_spec

# Function to load and preprocess audio files into Mel Spectrograms
def load_audio_file(file_path, max_length=None):
    waveform, sample_rate = torchaudio.load(file_path)
    
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_mels=20,  # Reduced number of Mel bins
        n_fft=512,  # Smaller FFT size
        hop_length=256,  # Smaller hop length
        win_length=512  # Smaller window size
    )
    
    mel_spec = transform(waveform)
    mel_spec = (mel_spec - mel_spec.mean()) / mel_spec.std()  # Normalize
    
    if max_length is not None:
        mel_spec = pad_or_truncate(mel_spec, max_length)
    
    mel_spec = mel_spec.unsqueeze(0)  # Add a batch dimension
    return mel_spec

import torch
import torch.nn as nn

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


# Apply pruning to the model to reduce the number of parameters
def prune_model(model):
    prune.random_unstructured(model.conv1, name="weight", amount=0.3)  # Prune 30% of weights
    prune.random_unstructured(model.conv2, name="weight", amount=0.3)
    prune.random_unstructured(model.fc1, name="weight", amount=0.3)
    prune.random_unstructured(model.fc2, name="weight", amount=0.3)
    return model

# Quantize the model for faster inference and smaller size
def quantize_model(model):
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)
    # Calibration step - pass some data through the model
    torch.quantization.convert(model, inplace=True)
    return model

def load_dataset(data_dir):
    file_paths = []
    labels = []
    
    # Assuming folder structure follows the UrbanSound8K format
    for fold_num in range(1, 11):  # 10 predefined folds (fold1 to fold10)
        fold_dir = os.path.join(data_dir, f'fold{fold_num}')
        for filename in os.listdir(fold_dir):
            if filename.endswith(".wav"):
                # Debugging: Print the filename and label extraction
                print(f"Processing file: {filename}")
                
                # Extract class label from the filename (e.g., '1' from 'filename-1.wav')
                try:
                    label = int(filename.split('-')[1])  # class IDs are 0-based
                    if label < 0 or label >= num_classes:
                        print(f"Warning: Invalid label {label} for file {filename}")
                    file_paths.append(os.path.join(fold_dir, filename))
                    labels.append(label)
                except ValueError:
                    print(f"Error: Failed to extract label from filename {filename}")
    
    return file_paths, labels


# Training and evaluation loop
def train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, max_length, device):
    model.train()
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            
            inputs_list = []
            targets_list = []  
            for idx in range(len(inputs)):
                mel_spec = load_audio_file(inputs[idx], max_length=max_length)  # Get mel spectrogram
                mel_spec = mel_spec.squeeze(0)  # Remove any singleton dimensions
                inputs_list.append(mel_spec)
                targets_list.append(targets[idx])
                
            data_tensor = torch.stack(inputs_list, dim=0).to(device)
            targets_tensor = torch.tensor(targets_list, dtype=torch.long).to(device)
            
            # Forward pass
            outputs = model(data_tensor)
            loss = criterion(outputs, targets_tensor)
            
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

    return model

# Cross-validation with predefined folds
def cross_validate(data_dir, device):
    print("Performing cross-validation...")
    file_paths, labels = load_dataset(data_dir)
    
    fold_accuracies = []

    for fold in range(1, 11):  # 10-fold cross-validation
        print(f"\nTraining fold {fold}")
        
        test_files = [f for i, f in enumerate(file_paths) if (i % 10) == fold - 1]
        test_labels = [labels[i] for i in range(len(file_paths)) if (i % 10) == fold - 1]
        train_files = [f for i, f in enumerate(file_paths) if (i % 10) != fold - 1]
        train_labels = [labels[i] for i in range(len(file_paths)) if (i % 10) != fold - 1]

        train_data = list(zip(train_files, train_labels))
        test_data = list(zip(test_files, test_labels))
        
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
        
        model = AudioClassifier().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        model = train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, max_length, device)
        
        model = prune_model(model)
        
        model_save_path = f"model_fold{fold}.pth"
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

    print("Cross-validation completed.")

if __name__ == "__main__":
    root_path = os.path.join('/Users', 'erik')
    data_path = os.path.join(root_path, 'sound_datasets', DATASET, 'audio')
    
    print("Initializing dataset...")
    dataset = soundata.initialize(DATASET)

    print("Beginning cross-validation...")
    cross_validate(data_path, device)
