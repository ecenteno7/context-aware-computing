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

# Hyperparameters
NUM_EPOCHS = 5  # Just for testing, can increase for proper training
BATCH_SIZE = 16  # For testing, smaller batch size works fine
LEARNING_RATE = 0.001
NUM_CLASSES = 10  # Adjust to your dataset
MAX_LENGTH = 1500  # Max length for audio processing
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET = 'urbansound8k'

# Function to pad or truncate the Mel spectrogram to a fixed length
def pad_or_truncate(mel_spec, max_length):
    length = mel_spec.size(2)  # Get the time dimension length (width)
    if length < max_length:
        # Pad with zeros if the length is shorter than max_length
        pad = max_length - length
        mel_spec = torch.nn.functional.pad(mel_spec, (0, pad))
    elif length > max_length:
        # Truncate if the length is longer than max_length
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

# Define the audio classifier model
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

# Load the dataset (for testing)
def load_dataset(data_dir):
    file_paths = []
    labels = []
    
    # Assuming folder structure follows the UrbanSound8K format
    for fold_num in range(1, 11):  # 10 predefined folds (fold1 to fold10)
        fold_dir = os.path.join(data_dir, f'fold{fold_num}')
        for filename in os.listdir(fold_dir):
            if filename.endswith(".wav"):
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

# Function to train the model
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        running_loss = 0.0
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

            running_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx+1}/{len(train_loader)}, Loss: {running_loss/10:.4f}")
                running_loss = 0.0

# Function to test the model
def test_model(model, test_loader, device):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs_list = []
            labels_list = []

            for idx in range(len(inputs)):
                mel_spec = load_audio_file(inputs[idx], max_length=max_length)  # Get mel spectrogram
                mel_spec = mel_spec.squeeze(0)  # Remove any singleton dimensions
                inputs_list.append(mel_spec)
                labels_list.append(labels[idx])
            
            data_tensor = torch.stack(inputs_list, dim=0).to(device)
            labels_tensor = torch.tensor(labels_list, dtype=torch.long).to(device)
            
            # Forward pass
            outputs = model(data_tensor)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels_tensor.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy

# Main function to train and test the model with cross-validation
def cross_validate(data_dir, device):
    file_paths, labels = load_dataset(data_dir)

    fold_accuracies = []

    for fold in range(1, 11):  # 10-fold cross-validation
        print(f"\nTraining fold {fold}")
        
        # Split data into train and test for this fold
        test_files = [f for i, f in enumerate(file_paths) if (i % 10) == fold - 1]
        test_labels = [labels[i] for i in range(len(file_paths)) if (i % 10) == fold - 1]
        train_files = [f for i, f in enumerate(file_paths) if (i % 10) != fold - 1]
        train_labels = [labels[i] for i in range(len(file_paths)) if (i % 10) != fold - 1]

        train_data = list(zip(train_files, train_labels))
        test_data = list(zip(test_files, test_labels))
        
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

        # Initialize and train the model
        model = AudioClassifier(num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Train the model
        train_model(model, train_loader, criterion, optimizer, device)
        
        # Test the model
        accuracy = test_model(model, test_loader, device)
        fold_accuracies.append(accuracy)
        print(f"Test Accuracy for fold {fold}: {accuracy * 100:.2f}%")

        # Save the model for this fold
        model_save_path = f"best_model_fold{fold}.pth"
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

    # Print overall cross-validation accuracy
    avg_accuracy = np.mean(fold_accuracies)
    print(f"\nAverage Accuracy across all folds: {avg_accuracy * 100:.2f}%")

if __name__ == "__main__":
    root_path = os.path.join('/Users', 'erik')
    data_path = os.path.join(root_path, 'sound_datasets', DATASET, 'audio')
    
    print("Initializing dataset...")
    dataset = soundata.initialize(DATASET)

    print("Beginning cross-validation...")
    cross_validate(data_path, device)
