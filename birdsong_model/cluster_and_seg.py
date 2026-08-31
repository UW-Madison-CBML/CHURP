import torch
from scipy.io import loadmat
import librosa
import pickle
import glob
import re
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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from sklearn.decomposition import PCA
import random
import pandas as pd
import sys
from scipy.signal import ellip, filtfilt
import soundfile as sf
import matplotlib.pyplot as plt
import seaborn
import sklearn
import iisignature
import umap
from sklearn.metrics import silhouette_score, pairwise_distances
from hdbscan import HDBSCAN
import base64
from IPython.display import display, HTML
import scipy.io.wavfile as wav
from matplotlib.gridspec import GridSpec
import math

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


#####################################
# Helper Functions
#####################################
def find_origin(spectrogram, embeddings_3d, perc=50, n=2):
    """
    Calculates a 3D fuzzy origin point by averaging the embeddings of the quietest 
    consecutive segments in an audio spectrogram.
    """
    # Calculate the average decibel level across all frequencies for each time frame
    avg_db = np.mean(spectrogram, axis=0)
    
    # Determine the decibel threshold at the specified percentile (default is the 50th percentile)
    db_thresh = np.percentile(avg_db, perc)
    
    # Create a boolean mask: True for 'loud' frames, False for 'quiet' frames 
    db_mask = avg_db > db_thresh 
    
    # Invert the mask (~db_mask) so True = quiet. 
    # Use a sliding window to find where 'n' consecutive quiet frames occur
    db_consecutive = np.convolve(~db_mask, np.ones(n), mode='valid')
    
    # Get the time indices where exactly 'n' consecutive quiet frames were detected
    keep_idx = np.where(db_consecutive == n)[0]
    
    # Extract the 3D embeddings for those quiet frames and average them together to find an origin
    origin = embeddings_3d[keep_idx].mean(axis=0)
    
    return origin, db_thresh, keep_idx
    
def get_all_intervals(embeddings_norm, origin, threshold_perc=0.2):
    """
    Identifies intervals/loops where the embeddings move a significant 
    distance away from the origin and then return.
    """
    # Calculate the Euclidean distance of each embedding frame from the origin
    distances = np.linalg.norm(embeddings_norm - origin, axis=1)
    
    # Define a distance threshold based on a percentage of the maximum distance found
    loop_threshold = distances.max() * threshold_perc
    
    # Create a mask for points that are far enough away to be considered part of a loop
    at_hub = distances > loop_threshold
    mask = at_hub 
    
    # Find the start indices: where the mask transitions from False to True
    starts = np.where(mask[1:] & (~mask[:-1]))[0] + 1
    
    # Find the end indices: where the mask transitions from True to False
    ends = np.where((~mask[1:]) & mask[:-1])[0] + 1
    
    # Edge case: If the sequence starts already outside the threshold, set the first start to index 0
    if mask[0]:
        starts = np.insert(starts, 0, 0)
        
    # Edge case: If the sequence ends while still outside the threshold, set the last end to the final index
    if mask[-1]:
        ends = np.append(ends, len(mask))
        
    # Combine start and end indices into paired tuples
    loop_intervals = list(zip(starts, ends))
    
    return loop_intervals, starts, ends

def refine_loops(starts, ends, min_length=20):
    """
    Filters out intervals that are shorter than a specified minimum length (in frames).
    """
    refined_intervals = []
    
    for s, e in zip(starts, ends):
        # Only keep intervals that last longer than min_length
        if (e - s) < min_length:
            continue
        else:
            refined_intervals.append((s, e))
            
    return refined_intervals


def get_all_subintervals(path, thresh_perc=0.75):
    """
    Takes a long embedding path and splits it into sub-loops by finding points 
    where the trajectory curves back near the starting point of the path.
    """
    # Treat the first point of this specific path as the local "hub"
    hub = path[0]
    
    # Calculate distance of all points in the path from this local hub
    distances = np.linalg.norm(path - hub, axis=1)
    threshold = distances.mean() * thresh_perc
    
    # Calculate the gradient (rate of change) of the distances
    slope = np.gradient(distances)
    
    # Find local minima: where the slope transitions from negative to positive/zero
    # This indicates the embedding was moving toward the hub, but is now moving away
    local_minima = (np.roll(slope, 1) < 0) & (slope >= 0) 
    
    # Only keep minima that also fall below the distance threshold
    minima_idx = np.where(local_minima & (distances < threshold))[0] 
    
    # Edge case handling for splitting the path based on these minima
    if len(minima_idx) == 0:
        pass
    elif minima_idx[-1] != len(distances):
        # Ensure the last subinterval goes all the way to the end of the path
        minima_idx = np.append(minima_idx, len(distances))
        
    # Create sub-intervals using adjacent pairs of minima indices
    loop_subintervals = list(zip(minima_idx[:-1], minima_idx[1:]))
    starts = minima_idx[:-1]
    ends = minima_idx[1:]
    
    return loop_subintervals, starts, ends


def collect_all_loops(embeddings, refined_intervals, thresh_perc=0.75, max_length = 50):
    """
    Iterates through broad intervals and conditionally splits long ones 
    into smaller sub-loops, packaging them into a structured dictionary.
    """
    # Unpack starts and ends, and calculate the duration of each interval
    starts, ends = map(np.array, zip(*refined_intervals))
    durations = ends - starts

    all_intervals = {}
    intervals = []

    for i, (d, (s, e)) in enumerate(zip(durations, refined_intervals)):
        # Extract the specific embedding slice for this loop
        path = embeddings[s:e]

        # If the loop is relatively short (<= 50 frames), keep it as is
        if d <= max_length:
            all_intervals[f"{i}_0"] = {'loop': path, 'idx': (s, e)}
            intervals.append((s, e))
        else:
            # If the loop is long, attempt to break it into sub-loops
            loop_subintervals, sub_starts, sub_ends = get_all_subintervals(path, thresh_perc=thresh_perc)
            refined_subintervals = refine_loops(sub_starts, sub_ends, min_length=10)

            # If it couldn't be split (or resulted in 1 piece), keep the original loop
            if len(refined_subintervals) <= 1:
                all_intervals[f"{i}_0"] = {'loop': path, 'idx': (s, e)}
                intervals.append((s, e))
            else: 
                # If it successfully split, add each sub-loop individually
                for j, (s_l, e_l) in enumerate(refined_subintervals):
                    # Offset the sub-loop indices by the global start index 's'
                    sub_path = embeddings[s+s_l:s+e_l]
                    key = f"{i}_{j}"
                    all_intervals[key] = {'loop': sub_path, 'idx': (s+s_l, s+e_l)}
                    intervals.append((s+s_l, s+e_l))

    return all_intervals, intervals


def get_loops_spec(spectrogram, intervals, t_sec, pad=20):
    """
    Extracts isolated spectrogram segments for each given interval and generates a plot figure
    """
    num_loops = len(intervals)
    
    # Calculate how many seconds each frame (point) represents
    sec_per_point = t_sec / spectrogram.shape[1]

    # Initialize a figure to plot all the isolated loops
    fig = plt.figure(figsize=(12, 3 * num_loops))
    segments = []   

    for i, (s, e) in enumerate(intervals):
        # Calculate padded start/end indices, ensuring they don't go out of array bounds
        s_pad = max(0, s - pad)
        e_pad = min(spectrogram.shape[1], e + pad)

        # Slice the spectrogram matrix to get just this segment
        segment = spectrogram[:, s_pad:e_pad]
        segments.append(segment)   

        # Calculate absolute time in seconds for the plot axes
        t_start = s_pad * sec_per_point
        t_end   = e_pad * sec_per_point

        # Add a subplot for this specific loop
        ax = fig.add_subplot(num_loops, 1, i + 1)
        ax.imshow(
            segment,
            origin='lower',
            aspect='auto',
            cmap='magma',
            extent=[t_start, t_end, 0, spectrogram.shape[0]]
        )

        ax.set_title(f"Syllable Spectrogram (Frames {s}-{e})")
        ax.set_ylabel("Freq Bin")
        
        # Only add the X-axis label to the very bottom plot to keep it clean
        if i == num_loops - 1:
            ax.set_xlabel("Time (s)")

    # Clean up layout and close the figure to prevent it from displaying immediately 
    plt.tight_layout()
    plt.close()

    return segments


def get_wav_segs(wav, spectrogram, intervals, t_sec):
    """
    Loads an audio file and extracts the raw audio waveforms corresponding 
    to specific time intervals (calculated via spectrogram frame counts).
    """
    waveform_segments = []
    
    # Load the audio file to get the raw waveform (y) and sampling rate (sr)
    file_path = wav
    y, sr = librosa.load(file_path)
    
    for inter in intervals:
        # Calculate time (in seconds) per spectrogram frame
        sec_per_point = t_sec / spectrogram.shape[1]
    
        # Add a tiny bit of padding (2 frames) to the start and end indices safely
        s_pad = max(0, inter[0] - 2)
        e_pad = min(spectrogram.shape[1], inter[1] + 2)

        # Convert padded frame indices to absolute time in seconds
        t_start = s_pad * sec_per_point
        t_end   = e_pad * sec_per_point
        
        # Convert absolute time in seconds to raw audio sample indices using the sampling rate
        start = round(t_start * sr)
        end = round(t_end * sr)

        # Slice out the raw audio segment
        waveform_segment = y[start:end]
        waveform_segments.append(waveform_segment)
        
    return waveform_segments

def validate_html(X, clusters, loop_specs, wav_segs, samplerate, all_intervals, num_records, output_file="output.html"):
    """
    Generates an interactive HTML report containing a 2D UMAP scatter plot 
    and a grid of audio players, spectrograms, and 3D trajectories for each cluster.
    """
    # Initialize the HTML structure and CSS styles for a responsive grid layout
    html_content = """
    <html>
    <head>
        <title>Syllable Clustering Results</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1, h2 { color: #333; }
            .grid-container { display: flex; flex-wrap: wrap; gap: 20px; }
            .grid-item { border: 1px solid #ccc; padding: 10px; border-radius: 5px; text-align: center; width: 22%; }
            audio { width: 100%; margin-bottom: 10px; }
            img { max-width: 100%; height: auto; margin-bottom: 10px; }
        </style>
    </head>
    <body>
        <h1>Clustering on Path Signature Embeddings</h1>
    """
    
    # Package data into a dictionary, keeping original indices to match 3D trajectories later
    syb_dict = {
        "cluster": clusters,
        "spectrogram": loop_specs,
        "waveform": wav_segs,
        "original_index": np.arange(len(clusters)), 
        "coords": list(X)
    }
    
    # Convert to DataFrame, shuffle to avoid displaying neighboring syllables, and sort by cluster ID to group them in the HTML
    syllable_dataframe = pd.DataFrame(syb_dict)
    syllable_dataframe = syllable_dataframe.sample(frac=1, ignore_index=True, random_state=seed)
    syllable_dataframe = syllable_dataframe.sort_values("cluster", axis=0, ascending=False)

    # 1. 2D UMAP PLOT
    fig2d = plt.figure(figsize=(10, 8))   
    ax2d = fig2d.add_subplot(111)

    is_noise = clusters == -1
    
    # Scatter plot of all valid (non-noise) points using UMAP coordinates
    ax2d.scatter(X[~is_noise, 0], X[~is_noise, 1], 
                 c=clusters[~is_noise], s=10, cmap='viridis', alpha=0.7, label='Clusters')
                     
    ax2d.set_title(f"HDBSCAN Clustering on Path Signatures (UMAP), {num_records} Recordings, {len(X)} Segements")
    ax2d.set_xlabel("UMAP 1") 
    ax2d.set_ylabel("UMAP 2") 

    #save as SVG
    buf2d = io.BytesIO()
    fig2d.savefig(buf2d, format='png', dpi=600, bbox_inches='tight')
    buf2d.seek(0)
    spec_str = base64.b64encode(buf2d.read()).decode('utf-8')
    plt.close(fig2d)

    html_content += f"<img src='data:image/png;base64,{spec_str}'>"

    # 2. SYLLABLE AUDIO, SPECTROGRAM & 3D TRAJECTORY GRID
    html_content += "<h2>Syllable Samples</h2><div class='grid-container'>"

    # Normalize the colormap so the 3D trajectory colors perfectly match the 2D UMAP plot
    valid_clusters = clusters[~is_noise]
    c_min = valid_clusters.min() if len(valid_clusters) > 0 else 0
    c_max = valid_clusters.max() if len(valid_clusters) > 0 else 1
    cmap = plt.get_cmap('viridis')

    # Reset tracker: display a maximum of 10 examples per cluster in the HTML grid
    num_seen = np.zeros(syllable_dataframe["cluster"].nunique() + 1)
    
    for _, sample in syllable_dataframe.iterrows():
        orig_idx = sample['original_index']
        c_id = sample['cluster']
        wave = sample["waveform"]
        spec = sample['spectrogram']
        traj = all_intervals[orig_idx] # Grab the specific 3D trajectory frames

        if num_seen[c_id] >= 10:
            continue
        else: 
            num_seen[c_id] += 1
        
        html_content += f"<div class='grid-item'><h3>Cluster: {c_id}</h3>"
        
        # --- Audio Player ---
        if wave is None or wave.size == 0:
            html_content += f"<p>Segment {orig_idx} is empty! Skipping...</p>"
        else:
            # Encode raw waveform data to a base64 .wav file and embed in HTML <audio> tag
            buf_audio = io.BytesIO()
            wav.write(buf_audio, samplerate, wave)
            buf_audio.seek(0)
            audio_str = base64.b64encode(buf_audio.read()).decode('utf-8')
            html_content += f"<audio controls><source src='data:audio/wav;base64,{audio_str}' type='audio/wav'></audio>"
        
        # --- Spectrogram Plot ---
        fig_spec = plt.figure(figsize=(4, 2))
        plt.imshow(spec, origin='lower', aspect='auto', cmap='magma')
        plt.title(f"Cluster {c_id} Spectrogram")
        plt.xlabel("Time (STFT bins)")
        plt.ylabel("Freq Bin")
        plt.tight_layout()
       
        #save as SVG
        buf_spec = io.BytesIO()
        fig_spec.savefig(buf_spec, format='png', dpi=600, bbox_inches='tight')
        buf_spec.seek(0)
        spec_str = base64.b64encode(buf_spec.read()).decode('utf-8')
        plt.close(fig_spec)

        # Update the HTML tag to image/png
        html_content += f"<img src='data:image/png;base64,{spec_str}'>"
        
        # --- 3D Individual Trajectory Plot ---
        fig3d = plt.figure(figsize=(4, 3))
        ax3d = fig3d.add_subplot(111, projection='3d')
        
        # Apply cluster color (or black if it's noise)
        if c_id == -1:
            color = 'black'
        else:
            norm_val = (c_id - c_min) / (c_max - c_min) if c_max > c_min else 0.5
            color = cmap(norm_val)
            
        # Draw the continuous trajectory line
        ax3d.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=color, alpha=0.8, linewidth=2)
        ax3d.set_title(f"Cluster {c_id} Trajectory", fontsize=12)

        # Draw intermediate points as gray dots
        ax3d.scatter(traj[1:-1, 0], traj[1:-1, 1], traj[1:-1, 2], 
                     color='gray', marker='.', s=15, alpha=0.6, zorder=4)

        # Draw start point as a green circle
        ax3d.scatter(traj[0, 0], traj[0, 1], traj[0, 2], 
                     color='green', marker='o', s=40, edgecolor='white', zorder=5)

        # Draw end point as a red X
        ax3d.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], 
                     color='red', marker='X', s=50, edgecolor='white', zorder=5)

        # Remove axes ticks for a cleaner embedded view
        ax3d.set_xticks([])
        ax3d.set_yticks([])
        ax3d.set_zticks([])
       
        #save as SVG
        buf3d = io.BytesIO()
        fig3d.savefig(buf3d, format='png', dpi=600, bbox_inches='tight')
        buf3d.seek(0)
        spec3d = base64.b64encode(buf3d.read()).decode('utf-8')
        plt.close(fig3d)

        # Update the HTML tag to image/png
        html_content += f"<img src='data:image/png;base64,{spec3d}'></div>"

    html_content += "</div></body></html>"
    
    # Write the complete HTML string to disk
    with open(output_file, "w", encoding='utf-8') as f:
        f.write(html_content)
        
    print(f"Successfully saved all visualizations and audio to {output_file}")

def annotate(ax, spec, extent, intervals, sec_per_point, colors=None, clusters=None, crop = False):
    """
    Plots a spectrogram on a given Matplotlib axis and adds colored
    highlight bars (annotations) just above the plot to indicate specific time intervals.
    """
    # Plot the 2D spectrogram array
    im = ax.imshow(spec, origin='lower', cmap='magma', aspect='auto', extent=extent)
    
    plot_start = extent[0]
    plot_end = extent[1]

    if crop:
        total_duration = plot_end - plot_start

        # Crop 25% off the beginning and 25% off the end of the entire spectrogram
        plot_start = plot_start + (0.25 * total_duration)
        plot_end = plot_end - (0.25 * total_duration)
        ax.set_xlim(plot_start, plot_end)

    # Iterate through each defined time interval
    for i, (start_idx, end_idx) in enumerate(intervals):
        # Determine the color for the annotation bar
        if colors is None:
            color = 'blue'  # Default color if no cluster information is provided
        else:
            # Assign color based on the cluster ID of this specific interval
            color = colors[clusters[i]]

        # Convert frame indices (start/end) to absolute time in seconds
        t_start_sec = start_idx * sec_per_point
        t_end_sec = (end_idx - 1) * sec_per_point


        # Skip this interval entirely if it is outside the cropped view 
        if t_end_sec < plot_start or t_start_sec > plot_end: 
            continue 

        # Clamp the edges so they don't bleed past the crop boundary 
        draw_start = max(plot_start, t_start_sec) 
        draw_end = min(plot_end, t_end_sec) 


        # Draw a horizontal colored bar to represent the interval.
        ax.axvspan(
            draw_start, draw_end,
            ymin=1.02, ymax=1.08,
            alpha=0.7, clip_on=False, label=f'Loop {i}',
            edgecolor='black', facecolor=color
        )
    
    ax.set_ylabel('Frequency')

    # Return the image object so a colorbar can be attached to it later
    return im

def plot_spectrograms(spectrogram, intervals, t_sec, save_name, colors, clusters):
    """
    Creates a wide-format figure showing a spectrogram with annotated time intervals,
    attaches a colorbar, and saves the final plot to disk.
    """
    # Calculate how many seconds each frame (point) on the x-axis represents
    sec_per_point = t_sec / spectrogram.shape[1]

    # set limits for plotting
    # Define middle half time bounds (25% to 75%)
    t_start = 0.25 * t_sec
    t_end = 0.75 * t_sec
    
    # Plot Spectrogram on primary y-axis
    img_extent = [0, t_sec, 0, spectrogram.shape[0]]
    im = ax1.imshow(spectrogram, origin='lower', aspect='auto', cmap='magma', extent=extent)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Frequency Bin")
    
    # Define the bounding box for the image data [x_min, x_max, y_min, y_max]
    img_extent = [0, t_sec, 0, spectrogram.shape[0]]

    # Initialize a large, wide figure (20x12)
    fig, ax = plt.subplots(1, 1, figsize=(20, 12), sharex=True)
    
    # Force the physical aspect ratio of the axes to be very wide and short (10:1 width-to-height)
    ax.set_box_aspect(0.2) 
    
    # Call the helper function to draw the spectrogram and the top-edge annotations
    im = annotate(ax, spectrogram, img_extent, intervals, sec_per_point, colors, clusters)
    
    # Add a colorbar mapped to the spectrogram's intensity (dB)
    cbar = plt.colorbar(im, ax=ax, label='Intensity (dB)', shrink=0.22, aspect=10, pad=0.02)
    
    ax.set_xlabel('Time (s)')

    ax.set_xlim(t_start, t_end)
    
    # Adjust layout so labels/colorbars aren't cut off during saving
    plt.tight_layout()
    
    # Save the figure to the provided filepath (e.g., .png or .pdf), keeping all edges tight
    plt.savefig(save_name, bbox_inches='tight')
    
    plt.close()

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

def optimize_umap_clusters(X, seed, min_cluster_size=100, max_clusters = None):
    """
    Sweeps UMAP n_components from 2 to 10, clusters with HDBSCAN, reassigns noise, 
    and evaluates via silhouette score. 
    """
    best_silhouette = -1.0
    best_clusters = None
    best_n = None
    X_2d = None


    for n_c in range(2, 11):

        # reset minimum cluster size for each dimension sweep
        temp_min_clust = min_cluster_size

        # Run UMAP
        reducer = umap.UMAP(n_components=n_c, metric='euclidean', random_state=seed)
        X_trans = reducer.fit_transform(X)
        
        if max_clusters is not None:
            while True: 
                # Run HDBSCAN
                hdbscan = HDBSCAN(min_cluster_size=temp_min_clust)
                clusters = hdbscan.fit_predict(X_trans)
                n_clusters = len(set(clusters) - {-1})

                # break loop if number of clusters (without counting noise) is less than max_clusters 
                if n_clusters <= max_clusters:
                    break
                else:
                    temp_min_clust = int(temp_min_clust * (n_clusters / max_clusters))  # Increase min_cluster_size to reduce number of clusters

        else:
            # Run HDBSCAN
            hdbscan = HDBSCAN(min_cluster_size=temp_min_clust)
            clusters = hdbscan.fit_predict(X_trans)

        # save 2d clusters and X-trans for fallback if no other dimension yields >1 cluster
        if n_c == 2:
            clusters_2d = clusters.copy()
            X_2d = X_trans.copy()
        
        # Force noise into nearest core cluster
        noise_mask = (clusters == -1)
        core_mask = ~noise_mask
        
        if np.any(noise_mask) and np.any(core_mask):
            X_noise = X_trans[noise_mask]
            X_core = X_trans[core_mask]
            core_clusters = clusters[core_mask]
            
            distances = pairwise_distances(X_noise, X_core, metric='euclidean')
            nearest_core_indices = np.argmin(distances, axis=1)
            
            # Overwrite the -1 noise labels
            clusters[noise_mask] = core_clusters[nearest_core_indices]
            
        # Evaluate (Silhouette score requires at least 2 distinct clusters)
        unique_clusters = np.unique(clusters)
        if len(unique_clusters) > 1:
            score = silhouette_score(X_trans, clusters)
            
            # Update best if current score is higher
            if score > best_silhouette:
                best_n = n_c
                best_silhouette = score
                best_clusters = clusters.copy()

    # Fallback in case NO dimension yielded >1 cluster
    if best_clusters is None:
        best_n = 2
        best_clusters = clusters_2d 
        
    print(f"Best Clustering is from UMAP with {best_n} components")

    return X_2d, best_clusters

def plot_distance_on_spectrogram(spectrogram, embeddings_3d, t_sec, origin, save_name):
    """
    Plots the latent trajectory distance from the origin overlaid on 
    the spectrogram and saves it as a 600 DPI PNG file.
    """
    distances = np.linalg.norm(embeddings_3d - origin, axis=1)
    
    fig, ax1 = plt.subplots(figsize=(14, 6))

    # set limits for plotting
    # Define middle half time bounds (25% to 75%)
    t_start = 0.25 * t_sec
    t_end = 0.75 * t_sec
    
    # Plot Spectrogram on primary y-axis
    extent = [0, t_sec, 0, spectrogram.shape[0]]
    im = ax1.imshow(spectrogram, origin='lower', aspect='auto', cmap='magma', extent=extent)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Frequency Bin")
    ax1.set_xlim(t_start, t_end)
    
    # Create twin axis for distance metric overlay
    ax2 = ax1.twinx()
    time_bins = np.linspace(0, t_sec, len(distances))
    
    # Plot distance line
    ax2.plot(time_bins, distances, color='cyan', linewidth=1.8, label='Distance from Origin')
    
    # Calculate mean distance and plot horizontal dashed red line
    mean_distance = np.mean(distances)
    ax2.axhline(y=mean_distance, color='red', linestyle='--', linewidth=1.5, label='Mean Distance')
    
    # Add legend for ax2 (will grab both labels automatically)
    ax2.legend(loc='upper right')
    
    ax2.set_ylabel("Distance from Origin")
    ax2.tick_params(axis='y')
    
    plt.title("Spectrogram with Latent Trajectory Distance Overlay")
    plt.tight_layout()
    plt.savefig(f"{save_name}.png", dpi=600, bbox_inches='tight')
    plt.close(fig)

def plot_large_latent_trajectory(embeddings_3d, save_name, figsize=(14, 12), birdname = None, recording_number = None):
    """
    Plots a large high-resolution 3D plot of the entire latent trajectory 
    of a recording in gray and saves it as a 600 DPI PNG.
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot continuous trajectory line in gray
    ax.plot(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2], 
            color='gray', alpha=0.7, linewidth=1.5)
    
    formatted_name = re.sub(r"([a-zA-Z]+)(\d+)", r"\1 \2", birdname)
    
    ax.set_title(f"Full Latent Trajectory, {formatted_name} Recording {recording_number}", fontsize=18)
    ax.set_xlabel("Principal Component 1", labelpad=10)
    ax.set_ylabel("Principal Component 2", labelpad=10)
    ax.set_zlabel("Principal Component 3", labelpad=10)
    
    plt.tight_layout()
    plt.savefig(f"{save_name}.png", dpi=600, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)

def plot_record_analysis(record, record_clusters, colors, save_name):
    """
    Takes a single record entry from record_df and plots:
    - The raw waveform
    - The absolute amplitude trace
    - The annotated spectrogram
    - A row for each cluster containing: 
        The 3D embeddings of the whole record (gray) with cluster segments highlighted
    """
    wav_path = record['wav_file']
    spec = record['spec']
    intervals = record['intervals']
    t_sec = record['t_sec']
    embeddings_3d = record['embeddings']
    
    # Skip if there are no intervals detected in this record
    if intervals is None or len(intervals) == 0:
        return
        
    # Load waveform and compute amplitude trace
    y, sr = librosa.load(wav_path)
    time_wav = np.linspace(0, t_sec, len(y))
    amplitude = np.abs(y) # Using absolute amplitude for the trace

    # set limits for plotting
    # Define middle half time bounds (25% to 75%)
    t_start = 0.25 * t_sec
    t_end = 0.75 * t_sec

    # Setup figure and GridSpec layout
    unique_clusters = sorted(list(set(record_clusters)))
    num_clusters = len(unique_clusters)
    
    # initialize first figure
    fig1 = plt.figure(figsize=(8.5, 5.5))
    gs1 = GridSpec(3, 4, figure=fig1, height_ratios=[1, 1, 2])

    # --- Row 0: Waveform ---
    ax_wav = fig1.add_subplot(gs1[0, :])
    ax_wav.plot(time_wav, y, color='black', lw=0.5)
    ax_wav.set_xlim(0, t_sec)
    ax_wav.set_title("Waveform")
    ax_wav.set_ylabel("Amplitude")
    ax_wav.set_xlim(t_start, t_end)
    ax_wav.set_xticks([]) # Hide x-ticks to share cleanly with spectrogram
    
    # --- Row 1: Amplitude Trace ---
    ax_amp = fig1.add_subplot(gs1[1, :], sharex=ax_wav)
    ax_amp.plot(time_wav, amplitude, color='forestgreen', lw=0.5)
    ax_amp.set_title("Amplitude Trace")
    ax_amp.set_ylabel("Abs Amp")
    ax_amp.set_xticks([])
    
    # --- Row 2: Annotated Spectrogram ---
    ax_spec = fig1.add_subplot(gs1[2, :], sharex=ax_wav)
    sec_per_point = t_sec / spec.shape[1]
    img_extent = [0, t_sec, 0, spec.shape[0]]
    annotate(ax_spec, spec, img_extent, intervals, sec_per_point, colors, record_clusters, crop = True)
    ax_spec.set_title("Annotated Spectrogram", pad = 20)
    ax_spec.set_xlabel("Time (s)")
    
    plt.tight_layout()
    plt.savefig(f"{save_name}_spec.png", bbox_inches='tight', dpi=600)
    plt.close(fig1)

    # initialize second figure
    fig2 = plt.figure(figsize=(8.5, 3 * num_clusters))
    gs2 = GridSpec(math.ceil(num_clusters / 3), 3, figure=fig2)

    # --- Lower Rows: Cluster-specific subplots ---
    for i, c_id in enumerate(unique_clusters):
        row_idx = i // 3
        col_idx = i % 3
        
        # Apply cluster color (or black for noise)
        color = colors[c_id] if c_id in colors else 'black'
        if c_id == -1: 
            color = 'black'

        # Find all intervals belonging to this specific cluster
        c_intervals = [inter for idx, inter in enumerate(intervals) if record_clusters[idx] == c_id]
                                        
        # Column: 3D Embeddings
        ax3d = fig2.add_subplot(gs2[row_idx, col_idx], projection='3d')

        # Plot the entire trajectory in translucent gray
        ax3d.plot(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2], color='gray', alpha=0.3, linewidth=1, zorder=1)
        
        # make the segments of the cluster colored by cluster
        for start, end in c_intervals:
            traj = embeddings_3d[start:end]
            ax3d.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=color, alpha=0.8, linewidth=1, zorder=10)
            
        ax3d.set_title(f"Cluster {c_id} Trajectories")
        ax3d.set_xticks([])
        ax3d.set_yticks([])
        ax3d.set_zticks([])

    plt.tight_layout()
    plt.savefig(f"{save_name}_clusters.png", bbox_inches='tight', dpi=600)
    plt.close(fig2)

####################################
# Main logic 
####################################
def main():
    # parse arguments
    parser = argparse.ArgumentParser(
        description="Arguments for birdsong segmentation and clustering."
    )
    parser.add_argument(
        "--pkl", 
        type=str, 
        help="The path to the pkl file from inference.")
    parser.add_argument(
        "--audio_path", 
        type=str, 
        help="The path to the wav files."
    )
    parser.add_argument(
        "--bird_name_prefix", 
        type=str, 
        help="The bird name for organizing results."
    )
    parser.add_argument(
        "--hop_length", 
        type=int, 
        help="The hop length of the stft."
    )
    parser.add_argument(
        "--max_length", 
        type=int, 
        help="The cutoff for subdivision of loops (in time bins)."
    )
    parser.add_argument(
        "--min_length", 
        type=int, 
        help="The shortest allowed loop length (in time bins)."
    )
    parser.add_argument(
        "--fst_threshold", 
        type=float, 
        help="The percentage of frequency intensity (as a fraction) to separate noise from song."
    )
    parser.add_argument(
        "--sec_threshold", 
        type=float, 
        help="The percentage of frequency intensity (as a fraction) to separate out syllables within song."
    )
    parser.add_argument(
        "--max_clusters", 
        type=int, 
        help="The maximum number of clusters allowed."
    )

    args = parser.parse_args()

    # load latent trajectories from inference
    with open(args.pkl, "rb") as file:
        data = pickle.load(file)
    
    # syllable-level data
    wav_segs = []
    loop_specs = []
    traj = []

    # recording-level data (for annotating)
    spectrograms = []
    intervals = []
    t_secs = []
    num_sybs = []
    wav_file = []
    spec_len = []
    embeddings = []
    
    # num syllables counter
    num = 0

    # store number of recordings
    num_records = len(data)

    # iterate over model output of each recording
    for index, (k, value) in enumerate(data.items()):

        embeddings_norm = value

        # temp directory for processing data
        data_dir = os.path.join(args.audio_path, k)
        name = os.path.basename(k).replace('.wav', '')

        # load spectrogram of sample
        spectrogram, t_sec, samplerate = load_stft(data_dir, args.hop_length)

        # find origin that represents silence
        origin, db_thresh, keep_idx = find_origin(spectrogram, embeddings_norm)

        if index % 10 == 0:
            # plot trajectory distance overlaid on spectrogram at 600 DPI
            dist_plot_name = f"dist_overlay_{args.bird_name_prefix}_{index}"
            plot_distance_on_spectrogram(spectrogram, embeddings_norm, t_sec, origin, dist_plot_name)

            # plot large version of entire latent trajectory in gray at 600 DPI
            large_traj_name = f"large_latent_traj_{args.bird_name_prefix}_{index}"
            plot_large_latent_trajectory(embeddings_norm, large_traj_name, birdname = args.bird_name_prefix, recording_number = index)

        # get intervals of deviation from fuzzy origin
        ints, starts, ends = get_all_intervals(embeddings_norm, origin, threshold_perc=args.fst_threshold)

        # filter out intervals that are too short to be syllables
        refined_intervals = refine_loops(starts, ends, min_length=args.min_length)

        # if no intervals pass the refining step, we store nothing in the corresponding data structures  
        if len(refined_intervals) == 0:
            # update recording-level info
            spectrograms.extend([spectrogram])
            intervals.extend([None])
            t_secs.extend([None])
            num_sybs.extend([0])
            spec_len.extend([spectrogram.shape[1]])
            wav_file.extend([data_dir])
            embeddings.extend([None])
            # update syllable-level info
            wav_segs.extend([None])
            loop_specs.extend([None])
            continue 
        else:
            # collect all smaller loops within the identified refined loops, using a higher threshold this time
            traj_small, intervals_small = collect_all_loops(embeddings_norm, refined_intervals, thresh_perc=args.sec_threshold, max_length = args.max_length)

            # store information of each syllable in appropriate data structure
            spec_ints = [item[1].get('idx') for item in traj_small.items()]
            loop_specs_small = get_loops_spec(spectrogram, spec_ints, t_sec, pad=0)
            wav_segs_small = get_wav_segs(data_dir, spectrogram, spec_ints, t_sec)
            # update recording-level info
            spectrograms.extend([spectrogram])
            intervals.extend([intervals_small])
            t_secs.extend([t_sec])
            num_sybs.extend([len(traj_small)])
            spec_len.extend([spectrogram.shape[1]])
            wav_file.extend([data_dir])
            embeddings.extend([embeddings_norm])
            # update syllable-level info
            wav_segs.extend(wav_segs_small)
            loop_specs.extend(loop_specs_small)
            for key, value in traj_small.items():
                traj.append(np.array(value.get('loop')))

    # make recording-level info into iterable data structure
    record_dict = {"spec": spectrograms,
                  "intervals": intervals,
                  "t_sec": t_secs,
                  "num": num_sybs,
                  "spec_len": spec_len,
                  "wav_file": wav_file,
                  "embeddings": embeddings}    
    record_df = pd.DataFrame(record_dict)

    # initialize embeddings of path signatures
    path_embeddings = []

    # loop through each syllable in the sample 
    for k in range(len(traj)):

        syb = traj[k]
        
        # Vectorized time-integration (MinMax scaled to [0, 1])
        num_rows = len(syb)
        time_steps = np.linspace(0, 1, num_rows)
        
        # add time dimension for time-integrated path signatures 
        transformed = np.c_[syb, time_steps]
        
        # calculate path signatures
        signature = iisignature.sig(transformed, 3)

        # add path signatures to embeddings list
        path_embeddings.append(signature)

    # turn path signatures into an array
    X = np.array(path_embeddings)

    # standard normalize values
    scaler = sklearn.preprocessing.StandardScaler()
    X = scaler.fit_transform(X)

    # run UMAP and HDBSCAN to cluster the path signatures, optimizing for silhouette score
    # the 2d umap will be used for plotting, but the cluster labels come from the optimal clustering
    X_2d, clusters = optimize_umap_clusters(X, seed=seed, min_cluster_size=100, max_clusters=args.max_clusters)

    # Create a dynamic color palette that scales based on the number of unique clusters found
    cmap = plt.get_cmap('viridis', len(set(clusters))) 
    colors = {}
    color_index = 0
    
    # Iterate through unique cluster IDs (sorted to ensure consistent color assignment)
    for cluster_id in sorted(set(clusters)): 
        # Assign a specific RGBA color tuple to each cluster ID
        colors[cluster_id] = cmap(color_index)
        color_index += 1

    # Export an interactive HTML file to visualize the embeddings, spectrograms, and raw audio.
    # We filter out any 'None' values from the specs and wav segments to prevent rendering errors for files without annotations
    validate_html(X_2d, clusters, 
                  [loop for loop in loop_specs if loop is not None], 
                  [seg for seg in wav_segs if seg is not None], 
                  samplerate, traj,
                  num_records, 
                  output_file=f"{args.bird_name_prefix}.html") 

    spec_num = 0
    
    # Iterate through the main dataframe containing all audio records
    for index, record in record_df.iterrows():
        # plot 1 out of every 50 spectrograms, and only if that recording actually contains detected intervals.
        if spec_num % 10 == 0 and record['intervals'] is not None:
            plot_spectrograms(record['spec'], record['intervals'], record['t_sec'], 
                              f"annotated_specs_{args.bird_name_prefix}_{spec_num}", 
                              colors, clusters) 
        spec_num = spec_num + 1

    # Convert the list of discrete intervals back into continuous, frame-by-frame label arrays 
    # for each original audio file.
    
    binned_predictions = {}
    wav_files = []
    annotated_times = []
    
    # 'cluster_idx' keeps track of the position in the flat 'clusters' array.
    # 'clusters' contains labels for all intervals across all files, squashed into 1D.
    cluster_idx = 0 
    
    for index, record in record_df.iterrows():
        wav_files.extend([record['wav_file']])
        
        # Initialize an array for the entire spectrogram length filled with -1 (background/silence)
        time_bins = np.full(record['spec_len'], -1)
        
        # If no bird syllables were detected in this file, store the background array and move on
        if record['intervals'] is None:
            annotated_times.extend([time_bins])
            continue

        # Grab the slice of cluster labels that belong ONLY to this specific record
        num_intervals = len(record['intervals'])
        record_clusters = clusters[cluster_idx : cluster_idx + num_intervals]
        
        # Call the figure 2 plotting function (for every 10 recordings)
        if index % 10 == 0:
            plot_name = f"record_analysis_{args.bird_name_prefix}_{index}"
            plot_record_analysis(record, record_clusters, colors, plot_name)
            
        # Iterate through the start/end frames of each detected syllable in this recording
        for i, (start, end) in enumerate(record['intervals']):
            # Overwrite the -1 background labels with the assigned cluster ID for these specific frames
            time_bins[start:end] = clusters[cluster_idx + i]
            
        annotated_times.extend([time_bins])
        
        # Advance the global cluster index tracker by the number of syllables found in this specific file
        cluster_idx += num_intervals
        
    # Package the filepaths and their corresponding frame-by-frame label arrays into a dictionary
    binned_predictions = {"wav_file": wav_files, "annotated_times": annotated_times}

    # pickle the dictionary so it can be analyzed later
    with open(f"annotated_bins_{args.bird_name_prefix}.pkl", "wb") as file:
        pickle.dump(binned_predictions, file)

if __name__ == "__main__":
     main()
