from nn_base.nn_base_data_loader import DataLoader 


class AudioDataLoader(DataLoader):
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
