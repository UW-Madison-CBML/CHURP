import torch
from scipy.io import loadmat
import pickle
import io
import os
os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/torch_cache" # Needed when running on CHTC
os.environ["USER"] = "researcher"
os.environ["LOGNAME"] = "researcher"

from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
import numpy as np
import random
import pandas as pd
import sys
import argparse

# Import the extracted loader and model classes
from model import SpectrogramLoader, pad_collate_fn, ConvAutoencoderLSTM

# set random seeds for reproducability
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

#############################################
#  Training
#############################################
def main():
    # parse arguments
    parser = argparse.ArgumentParser(
        description="Arguments for birdsong training."
    )
    parser.add_argument(
        "--audio_path", 
        type=str, 
        help="The path to the wav files."
    )
    parser.add_argument(
        "--save_name", 
        type=str, 
        help="The file name to save the model weights."
    )
    parser.add_argument(
        "--hop_length", 
        type=int, 
        help="The hop length of the stft."
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        help="The number of training epochs"
    )
    args = parser.parse_args()

    # load wav files as spectrograms
    dataset = SpectrogramLoader(args.audio_path, args.hop_length)

    # preprocess and bactch data  
    dataloader = DataLoader(dataset, batch_size=24, shuffle=True, collate_fn=pad_collate_fn)

    # empty cache and use GPU
    torch.cuda.empty_cache()
    device = torch.device("cuda")

    # intialize variables for training 
    model = ConvAutoencoderLSTM().to(device)
    # do not aggregate loss across batch (reduction = 'none')
    criterion = nn.MSELoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # train model using mask to ignore loss from padded regions 
    n_epochs = args.epochs
    model.train()
    for epoch in range(n_epochs):
        # Unpack the mask alongside the batch
        for batch, mask, lengths, filenames in dataloader:
            batch = batch.to(device)
            
            # Expand mask from (B, T) to (B, 1, T) so it broadcasts over the 512 freq bins
            mask = mask.unsqueeze(1).to(device)
            
            output, latent = model(batch)
            
            # Calculate raw element-wise loss (Shape: B, 512, T)
            unreduced_loss = criterion(output, batch)
            
            # Multiply by the mask to zero out the padded regions
            masked_loss = unreduced_loss * mask
            
            # Calculate the mean only over the valid elements
            # Multiply mask.sum() by 512 (frequency bins) to get total valid elements
            valid_elements = mask.sum() * 512
            loss = masked_loss.sum() / valid_elements
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print(f"Epoch {epoch+1}, Loss: {loss.item():.10f}")

    # save model weights
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, args.save_name)

if __name__ == "__main__":
    main()
