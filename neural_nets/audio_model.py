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


class MelSpectrogramCNN(nn.Module):
    def __init__(self, num_classes=10, conv_config=None):
        super(AudioClassifier, self).__init__()

        if conv_config is None:
            # Default configuration: [(channels, kernel_size, stride), ...]
            conv_config = [
                (16, 3, 1),  # Overlapping in early layers
                (32, 3, 2),  # Moderate overlap, start downsampling
                (64, 5, 2),  # Non-overlapping, focus on global features
            ]

        layers = []
        in_channels = 1  # Starting with 1-channel input (Mel spectrogram)

        for out_channels, kernel_size, stride in conv_config:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size // 2))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))  # Optional pooling
            in_channels = out_channels

        self.feature_extractor = nn.Sequential(*layers)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 3 * 3, 256)  # Adjust dimensions dynamically
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.flatten(x)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class AudioClassifier(BaseModel):
    def __init__(self, config, dataset):
        super.__init__(config, dataset)
        self.define_model()
        self.cross_validate() 

    def define_model(self, kernel_size=2, stride=2):
        self.model = MelSpectrogramCNN()
        return 

    def compile_model(self):
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        return

    def fit_model(self):
        self.model.train()

        for epoch in range(self.config.num_epochs):
            print(f"Epoch {epoch+1}/{self.config.num_epochs}")
            
            running_loss = 0.0
            for batch_idx, (inputs, targets) in enumerate(self.dataset.train_data):
                self.optimizer.zero_grad()
                
                inputs_list = []
                targets_list = []  
                for idx in range(len(inputs)):
                    mel_spec = self.dataset.load_audio_file(inputs[idx], max_length=max_length)  # Get mel spectrogram
                    mel_spec = mel_spec.squeeze(0)  # Remove any singleton dimensions
                    inputs_list.append(mel_spec)
                    targets_list.append(targets[idx])
                    
                data_tensor = torch.stack(inputs_list, dim=0).to(device)
                targets_tensor = torch.tensor(targets_list, dtype=torch.long).to(device)
                
                # Forward pass
                outputs = self.model(data_tensor)
                loss = self.loss_function(outputs, targets_tensor)
                
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if batch_idx % 10 == 0:
                    print(f"Batch {batch_idx+1}/{len(train_loader)}, Loss: {running_loss/10:.4f}")
                    running_loss = 0.0

    def evaluate_model(self):
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in self.dataset.test_data:
                inputs_list = []
                labels_list = []

                for idx in range(len(inputs)):
                    mel_spec = self.dataset.load_audio_file(inputs[idx], max_length=max_length)  # Get mel spectrogram
                    mel_spec = mel_spec.squeeze(0)  # Remove any singleton dimensions
                    inputs_list.append(mel_spec)
                    labels_list.append(labels[idx])
                
                data_tensor = torch.stack(inputs_list, dim=0).to(self.device)
                labels_tensor = torch.tensor(labels_list, dtype=torch.long).to(self.device)
                
                # Forward pass
                outputs = self.model(data_tensor)
                _, predicted = torch.max(outputs, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels_tensor.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_preds)
        return accuracy

    # Main function to train and test the model with cross-validation
    def cross_validate(self):
        file_paths, labels = load_dataset(self.data_dir)

        fold_accuracies = []

        for fold in range(1, 11):  # 10-fold cross-validation
            print(f"\nTraining fold {fold}")
            
            # Split data into train and test for this fold
            test_files = [f for i, f in enumerate(self.dataset) if (i % 10) == fold - 1]
            test_labels = [labels[i] for i in range(len(self.dataset)) if (i % 10) == fold - 1]
            train_files = [f for i, f in enumerate(self.dataset) if (i % 10) != fold - 1]
            train_labels = [labels[i] for i in range(len(self.dataset)) if (i % 10) != fold - 1]

            train_data = list(zip(train_files, train_labels))
            test_data = list(zip(test_files, test_labels))
            
            self.dataset.train_data = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
            self.dataset.test_data = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

            # Train the model
            self.fit_model()
            
            # Test the model
            accuracy = self.evaluate_model()
            fold_accuracies.append(accuracy)
            print(f"Test Accuracy for fold {fold}: {accuracy * 100:.2f}%")

            # Save the model for this fold
            model_save_path = f"best_model_fold{fold}.pth"
            torch.save(self.model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")

        # Print overall cross-validation accuracy
        avg_accuracy = np.mean(fold_accuracies)
        return
        
    def predict(self):
        pass

