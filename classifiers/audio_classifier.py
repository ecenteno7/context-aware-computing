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
import torchaudio

# import scripts from other folders
import os
import sys
# sys.path.append('../')
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from nn_base.base_neural_nets import BaseNeuralNet


class MelSpectrogramCNN(nn.Module):
    def __init__(self, num_classes=10, conv_config=None, input_shape=(1, 20, 1500)):
        super().__init__()
        
        if conv_config is None:
            conv_config = [
                (16, (3, 3), (1, 1)),  # Conv layer 1
                (32, (3, 3), (2, 2)),  # Conv layer 2
                (64, (3, 3), (2, 2)),  # Conv layer 3
            ]
        
        layers = []
        in_channels = 1  # Starting with single-channel input
        
        for out_channels, kernel_size, stride in conv_config:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size[0] // 2))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
            in_channels = out_channels

        self.feature_extractor = nn.Sequential(*layers)
        self.flatten = nn.Flatten()

        # Dynamically calculate feature size
        with torch.no_grad():
            dummy_input = torch.zeros((1, *input_shape))  # Match actual input size
            feature_size = self.feature_extractor(dummy_input).numel()

        self.fc1 = nn.Linear(feature_size, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x




class AudioClassifier(BaseNeuralNet):
    def __init__(self, config, dataset, load=False):
        super().__init__(config, dataset)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.define_model()
        if load:
            self.model.load_state_dict(torch.load(self.saved_model_path, weights_only=True))
            return     
        self.compile_model()
        self.cross_validate() 
    
    def define_model(self, kernel_size=2, stride=2):
        self.model = MelSpectrogramCNN()
        self.saved_model_path = os.path.join(self.config.config_namespace.saved_model_dir, 'audio_classifier.pth')
        self.model.to(self.device)
        return 

    def compile_model(self):
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        return

    def fit_model(self):
        self.model.train()

        for epoch in range(self.config.config_namespace.num_epochs):
            print(f"Epoch {epoch+1}/{self.config.config_namespace.num_epochs}")
            
            running_loss = 0.0
            for batch_idx, (inputs, targets) in enumerate(self.dataset.train_data):
                self.optimizer.zero_grad()
                inputs_list = []
                targets_list = []  
                for idx in range(len(inputs)):
                    mel_spec = self.dataset.load_audio_file(inputs[idx], max_length=1500)  # Get mel spectrogram
                    mel_spec = mel_spec.squeeze(0)  # Remove any singleton dimensions
                    inputs_list.append(mel_spec)
                    targets_list.append(targets[idx])
                    
                data_tensor = torch.stack(inputs_list, dim=0).to(self.device)
                targets_tensor = torch.tensor(targets_list, dtype=torch.long).to(self.device)
                
                # Forward pass
                outputs = self.model(data_tensor)
                loss = self.loss_function(outputs, targets_tensor)
                
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if batch_idx % 10 == 0:
                    print(f"Batch {batch_idx+1}/{len(self.dataset.train_data)}, Loss: {running_loss/10:.4f}")
                    running_loss = 0.0

    def evaluate_model(self):
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in self.dataset.test_data:
                inputs_list = []
                labels_list = []

                for idx in range(len(inputs)):
                    mel_spec = self.dataset.load_audio_file(inputs[idx], max_length=1500)  # Get mel spectrogram
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
        fold_accuracies = []
        for fold_num in range(1, 11):  # 10-fold cross-validation
            fold = f'fold{fold_num}/'

            print(f"\nTraining {fold}")
                        
            # Split data into train and test for this fold
            train_files = [f for i, f in enumerate(self.dataset.data) if fold not in f]
            test_files = [f for i, f in enumerate(self.dataset.data) if fold in f]
            train_labels = [self.dataset.labels[i] for i, f in enumerate(self.dataset.data) if fold not in f]
            test_labels = [self.dataset.labels[i] for i, f in enumerate(self.dataset.data) if fold in f]
            print(len(train_files))
            print(len(test_files))
            train_data = list(zip(train_files, train_labels))
            test_data = list(zip(test_files, test_labels))
            self.dataset.train_data = torch.utils.data.DataLoader(train_data, batch_size=10, shuffle=True)
            self.dataset.test_data = torch.utils.data.DataLoader(test_data, batch_size=10, shuffle=False)

            # Train the model
            self.fit_model()
            
            # Test the model
            accuracy = self.evaluate_model()
            fold_accuracies.append(accuracy)
            print(f"Test Accuracy for fold {fold}: {accuracy * 100:.2f}%")

            # Save the model for this fold
            model_save_path = os.path.join(self.config.config_namespace.saved_model_dir, f"model_fold{fold_num}.pth")
            torch.save(self.model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")

        # Print overall cross-validation accuracy
        avg_accuracy = np.mean(fold_accuracies)
        return

    def predict(self, input_file_path):
        self.model.eval()
        
        input = self.dataset.load_audio_file(input_file_path, max_length=1500)
        outputs = self.model(input.to(self.device))
        _, predicted = torch.max(outputs, 1)

        predicted_label = self.config.config_namespace.class_names[predicted[0]]
        
        return predicted_label
