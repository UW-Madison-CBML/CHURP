import torch
from scipy.io import loadmat
import librosa
import pickle
import glob
import argparse
import io
import os
import torchaudio
os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/torch_cache" # Needed when running on CHTC
os.environ["USER"] = "researcher"
os.environ["LOGNAME"] = "researcher"
import torchaudio.transforms as T
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader, random_split
import numpy as np
from sklearn.decomposition import PCA
import random
import pandas as pd
# from sklearn.preprocessing import LabelEncoder
import sys
from scipy.signal import ellip, filtfilt
import soundfile as sf
import matplotlib.pyplot as plt
import seaborn
import sklearn
import iisignature
import umap
from hdbscan import HDBSCAN
import base64
from IPython.display import display, HTML
import scipy.io.wavfile as wav

# Import the extracted loader, collate function, and model classes
from model import SpectrogramLoader, pad_collate_fn, ConvAutoencoderLSTM

#increase global font sizes
plt.rcParams.update({
    'font.size': 14,          # Global font size
    'axes.titlesize': 16,     # Title size
    'axes.labelsize': 14,     # X and Y label size
    'xtick.labelsize': 12,    # X tick label size
    'ytick.labelsize': 12     # Y tick label size
})

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

    # Force deterministic algorithms in cuDNN
    torch.backends.cudnn.deterministic = True
        
    # Disable benchmarking (which selects the fastest algorithm dynamically)
    torch.backends.cudnn.benchmark = False


######################################################################
# Helper funtions for inference 
######################################################################
def load_stft(f, hop_length):
    """
    Helper for loading individual wav files as spectrograms -- normalizing and cutoff at 512 bins not needed here
    """
    # load file as single-channel audio
    with sf.SoundFile(f, 'r') as wav_file:
        samplerate = wav_file.samplerate
        total_frames = wav_file.frames
        data = wav_file.read(dtype='int16')

    # apply high-pass filter with 500hz cutoff frequency (filter forwards and backwards)
    b, a = ellip(5, 0.2, 40, 500 / (samplerate / 2), 'high')
    data = filtfilt(b, a, data)

    # transform audio to spectrogram with short-time fourier
    Sxx = librosa.stft(data.astype(float), n_fft=1024, hop_length=hop_length, window='hann')
    Sxx_log = librosa.amplitude_to_db(np.abs(Sxx), ref=np.max)
    
    t_sec = total_frames / samplerate

    return Sxx_log, t_sec, samplerate

######################################################################
# Main Inference Logic
######################################################################
def main():
    # parse arguments
    parser = argparse.ArgumentParser(
        description="Arguments for birdsong inference."
    )
    parser.add_argument(
        "--audio_path", 
        type=str, 
        help="The path to the wav files."
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        help="The path of the model weights file."
    )
    parser.add_argument(
        "--save_pickle", 
        type=str, 
        help="The file name to save the inference outputs."
    )
    parser.add_argument(
        "--hop_length", 
        type=int, 
        help="The hop length of the stft."
    )
    args = parser.parse_args()

    # load wav files as spectrograms
    dataset = SpectrogramLoader(args.audio_path, args.hop_length)
    
    # preprocess and batch data
    dataloader = DataLoader(dataset, batch_size=24, shuffle=False, collate_fn=pad_collate_fn)

    # empty cache and use GPU
    torch.cuda.empty_cache()
    device = torch.device("cuda")

    # Initialize and load the model
    model = ConvAutoencoderLSTM().to(device)
    checkpoint = torch.load(args.model_path, map_location=torch.device('cuda'))
    model.load_state_dict(checkpoint['model_state_dict'])

    # initialize dictionary for storing inference results: latent and reconstruction of each recording
    data = {}
    raw_latents = []

    # run inference on each batch of recordings
    model.eval()
    with torch.no_grad():
        for batch_specs, mask, lengths, batch_filenames in dataloader:
            batch_specs = batch_specs.to(device)
            recon, latent = model(batch_specs)

            # Iterate through the batch and slice off the padding
            for i in range(recon.size(0)):
                actual_length = lengths[i]
                valid_reconstruction = recon[i, :, :actual_length]
                valid_latent = latent[i, :actual_length, :]
                name = batch_filenames[i]
                data[name] = {
                    "latent": valid_latent.cpu().numpy(),
                    "recon": valid_reconstruction.cpu().numpy()
                }
                raw_latents.extend(valid_latent.cpu().numpy())

    # learn data-wide parameters for PCA and normalization so that are the same across multiple recordings
    pca = PCA(n_components=3, random_state=seed)
    latent_reduced = pca.fit_transform(np.vstack(raw_latents))
    min_val = np.min(latent_reduced, axis=0)
    max_val = np.max(latent_reduced, axis=0)

    # intialize dictionary for storing normalized 3D embeddings of each recording
    embeddings_3d_norm = {}

    # iterate over model output of each recording
    for k, value in data.items():

        # temp directory for processing data
        data_dir = os.path.join(args.audio_path, k)
        name = os.path.basename(k).replace('.wav', '')

        # load spectrogram of sample
        spectrogram, t_sec, samplerate = load_stft(data_dir, args.hop_length)

        # reduce to 3d (using global components)
        embeddings_3d = pca.transform(value['latent'])

        # min-max normalize 3d points
        embeddings_norm = (embeddings_3d - min_val) / (max_val - min_val + 1e-10)

        embeddings_3d_norm[k] = embeddings_norm

    with open(args.save_pickle, "wb") as file:
        pickle.dump(embeddings_3d_norm, file)

if __name__ == "__main__":
    main()
