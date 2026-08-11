import torch
import librosa
import glob
import os
from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np
from scipy.signal import ellip, filtfilt
import soundfile as sf
import torch.nn.functional as F

#########################################
# Data Loader and Model Definition
#########################################
class SpectrogramLoader(Dataset):
    def __init__(self, wav_dir, hop):

        # find all wav files in the directory, sorted for reproducability
        self.file_list = sorted(glob.glob(f"{wav_dir}/*wav"))

        # store hop length
        self.hop_length = hop

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file = self.file_list[idx]

        # load each item as single-channel audio
        with sf.SoundFile(file, 'r') as wav_file:
            samplerate = wav_file.samplerate
            data = wav_file.read(dtype='int16')

        # apply high-pass filter with 500hz cutoff frequency (filter forwards and backwards)
        b, a = ellip(5, 0.2, 40, 500 / (samplerate / 2), 'high')
        data = filtfilt(b, a, data)

        # transform audio to spectrogram with short-time fourier
        Sxx = librosa.stft(data.astype(float), n_fft=1024, hop_length=self.hop_length, window='hann')
        Sxx_log = librosa.amplitude_to_db(np.abs(Sxx), ref=np.max)

        # min-max normalize decible values from [-80,0] to [0,1]
        spec_min = -80
        spec_max = 0
        normalized = (Sxx_log - spec_min) / (spec_max - spec_min)

        # remove last frequency bin to go 513->512 for even number, but leave the time axis (axis 1) variable-length
        normalized = normalized[:512, :]

        # return filename -- only important for inference
        filename = os.path.basename(file)

        return torch.as_tensor(normalized, dtype=torch.float32), filename 

def pad_collate_fn(batch):
    # batch is a list of tuples: [(tensor1, filename1), (tensor2, filename2), ...]
    
    # Use item[0] to get the tensor's shape
    max_len = max(item[0].shape[1] for item in batch)
    padded_batch = []
    masks = []
    lengths = [] # Track original lengths for cropping padding
    filenames = [item[1] for item in batch]

    for item in batch:
        x = item[0] # Extract the tensor from the tuple
        seq_len = x.shape[1]
        
        pad_amount = max_len - seq_len
        padded_x = F.pad(x, (0, pad_amount), mode='constant', value=0.0)
        padded_batch.append(padded_x)

        # create a mask that = 0 for the padded region -- will be used for loss calculation
        mask = torch.zeros(max_len)
        mask[:seq_len] = 1.0
        masks.append(mask)
        
    # Return lengths alongside batch and masks
    return torch.stack(padded_batch, dim=0), torch.stack(masks, dim=0), lengths, filenames

class ConvAutoencoderLSTM(nn.Module):
    def __init__(
            self,
            input_channels=512,
            kernel_size=3,
            num_layers=1,
            lstm_hidden_channels=4096,
    ):
        # model architecture begins with 2 convolutional layers
        super(ConvAutoencoderLSTM, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=kernel_size, stride=(2, 1), padding=1), 
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=kernel_size, stride=(2, 1), padding=1),
            nn.ReLU()
        )

        # feature dimension calcualtions for when convolution outputs are flattened
        reduced_freq = input_channels // 4 
        self.feature_dim = 32 * reduced_freq #(32 * 128 = 4096)

        # linear layer for the convolution outputs 
        self.compressor = nn.Linear(self.feature_dim, lstm_hidden_channels) 

        # lstm to learn latent representation
        self.lstm = nn.LSTM(
            input_size=lstm_hidden_channels,   
            hidden_size=lstm_hidden_channels,
            num_layers=num_layers,
            batch_first=True #(batch, seq, feature) 
        )

        # linear later to process latent
        self.expander = nn.Linear(lstm_hidden_channels, self.feature_dim)

        # 2 transpose convolutional layers to decode latent representation
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=kernel_size, stride=(2, 1), padding=1, output_padding=(1, 0)), #padding=1
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=kernel_size, stride=(2, 1), padding=1, output_padding=(1, 0)),
            nn.Sigmoid() 
        )

    def forward(self, x):

        # foward pass, see comments for dimensions
        x = x.unsqueeze(1)
        B, C, F, T = x.shape # (B, 1, 512, T)

        x = self.encoder(x) # (B, 32, 128, T)
 
        x = x.permute(0, 3, 1, 2).contiguous() # (B, T, 32, 128)
        x = x.view(B, T, -1) # (B, T, 4096)
        
        compressed = self.compressor(x) # (B, 4096, 4096)
        lstm_out, _ = self.lstm(compressed) # (B, 4096, 4096)
        expanded = self.expander(lstm_out) # (B, 4096, 4096)
        
        x = expanded.view(B, T, 32, -1) # (B, T, 32, 128)
        x = x.permute(0, 2, 3, 1).contiguous() # (B, 32, 128, T)
    
        reconstruction = self.decoder(x) # (B, 1, 512, T)

        final = reconstruction.squeeze(1) # (B, 512, T)

        return final, lstm_out
