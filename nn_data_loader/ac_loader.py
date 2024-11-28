
# import scripts from other folders
import os
import sys
# sys.path.append('../')
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from nn_base.nn_base_data_loader import DataLoader 
import torchaudio
import torch
import torchaudio.transforms as T
import numpy as np

class FoldedAudioDataLoader(DataLoader):
    def __init__(self, config):
        super().__init__(config)


    def load_dataset(self, folds=10):
        file_paths = np.array([])
        labels = np.array([])
        
        # Assuming folder structure follows the UrbanSound8K format
        for fold_num in range(1, folds + 1):  # 10 predefined folds (fold1 to fold10)
            fold_dir = os.path.join(self.config.config_namespace['dataset_dir'], f'fold{fold_num}')
            for filename in os.listdir(fold_dir):
                if filename.endswith(".wav"):
                    # Extract class label from the filename (e.g., '1' from 'filename-1.wav')
                    try:
                        label = int(filename.split('-')[1])  # class IDs are 0-based
                        if label < 0 or label >= 10:
                            print(f"Warning: Invalid label {label} for file {filename}")
                        file_paths = np.append(file_paths, os.path.join(fold_dir, filename))
                        labels = np.append(labels, label)
                    except ValueError:
                        print(f"Error: Failed to extract label from filename {filename}")

        self.data = file_paths
        self.labels = labels
        return

    def display_data_element(self, which_data, index):
        print(which_data[index])

    def preprocess_dataset(self):
        return

    # Function to pad or truncate the Mel spectrogram to a fixed length
    def pad_or_truncate(self, mel_spec, max_length):
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
    def load_audio_file(self, file_path, max_length=None):
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
            mel_spec = self.pad_or_truncate(mel_spec, max_length)

        mel_spec = mel_spec.unsqueeze(0)  # Add a batch dimension
        return mel_spec
