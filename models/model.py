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
        
        compressed = self.compressor(x) # (B, T, 4096)
        lstm_out, _ = self.lstm(compressed) # (B, T, 4096)
        expanded = self.expander(lstm_out) # (B, T, 4096)
        
        x = expanded.view(B, T, 32, -1) # (B, T, 32, 128)
        x = x.permute(0, 2, 3, 1).contiguous() # (B, 32, 128, T)
    
        reconstruction = self.decoder(x) # (B, 1, 512, T)

        final = reconstruction.squeeze(1) # (B, 512, T)

        return final, lstm_out
