import argparse
import os
import pickle
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import ellip, filtfilt

def load_stft(f, hop_length):
    """
    Loads individual wav files as spectrograms.
    """
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

    # calculate the length of the recording in seconds
    t_sec = total_frames / samplerate

    return Sxx_log, t_sec, samplerate

def convert_label(val):
    """
    Converts single alphabetic characters ('a' -> '10', 'b' -> '11', etc.)
    while leaving standard numbers unchanged.
    """
    s = str(val).strip().lower()
    if len(s) == 1 and 'a' <= s <= 'z':
        return str(ord(s) - ord('a') + 10)
    return str(val)


def annotate(ax, spec, extent, intervals, t_sec, samplerate, colors=None, clusters=None, hop=None):
    """
    Plots a spectrogram on a given Matplotlib axis and adds colored
    highlight bars (annotations) just above the plot to indicate specific time intervals.
    """
    im = ax.imshow(spec, origin='lower', cmap='magma', aspect='auto', extent=extent)

    sec_per_bin = (1 / samplerate)

    # Define your zoom limits -- displays middle 50% of spectrogram where audio typically is 
    x_min, x_max = t_sec * 0.25, t_sec * 0.75

    for i, (start_idx, end_idx) in enumerate(intervals):
        color = 'blue' if colors is None else colors[clusters[i]]

        # Convert indices to absolute time
        t_start_sec = start_idx * sec_per_bin
        t_end_sec = end_idx * sec_per_bin

        # Manually clip the start and end times to the x-limits so they don't bleed horizontally when clip_on=False
        draw_start = max(t_start_sec, x_min)
        draw_end = min(t_end_sec, x_max)

        # Only draw the span if it actually falls within the zoomed window
        if draw_start < draw_end:
            ax.axvspan(
                draw_start, draw_end,
                ymin=1.02, ymax=1.08,
                alpha=0.7, clip_on=False, label=f'Loop {i}',
                edgecolor='black', facecolor=color
            )

    ax.set_xlim(x_min, x_max)
    ax.set_ylabel('Frequency')

    return im

def plot_spectrograms(spectrogram, intervals, t_sec, samplerate, save_name, colors, clusters, hop):
    """
    Creates a wide-format figure showing a spectrogram with annotated time intervals,
    attaches a colorbar, and saves the final plot to disk.
    """
    img_extent = [0, t_sec, 0, spectrogram.shape[0]]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6), sharex=True)
    ax.set_box_aspect(0.2)
    
    # annotate spectrogram with cluster labels
    im = annotate(ax, spectrogram, img_extent, intervals, t_sec, samplerate, colors, clusters, hop)
    
    cbar = plt.colorbar(im, ax=ax, label='Intensity (dB)', shrink=0.22, aspect=10, pad=0.02)
    ax.set_xlabel('Time (s)')

    # only show middle 50% of spectrogram where audio generally occurs
    ax.set_xlim(t_sec * 0.25, t_sec * 0.75)
    plt.tight_layout()
    plt.savefig(save_name, format='png', dpi=600, bbox_inches='tight')
    plt.close()

def extract_intervals_and_clusters(annot_array, bg_labels=("-1", -1)):
    """
    Converts a continuous array of frame-wise annotations into bounding intervals and cluster IDs.
    Collapses continuous identical labels into discrete intervals.
    Safely handles string-based labels to support XML formats.
    """
    intervals = []
    clusters = []
    
    if not annot_array:
        return intervals, clusters

    # Force background labels to strings for safe comparison
    bg_labels_str = {str(b) for b in bg_labels}
    
    start_idx = 0
    current_label = str(annot_array[0])

    # loop through each frame and start/stop intervals at label transitions
    for i in range(1, len(annot_array)):
        label = convert_label(annot_array[i])       
        if label != current_label:
            if current_label not in bg_labels_str:
                intervals.append((start_idx, i))
                # Store as string to avoid ValueErrors on XML labels like 'a', 'b', 'syl1'
                clusters.append(current_label) 
                
            current_label = label
            start_idx = i
            
    # Handle the final run
    if current_label not in bg_labels_str:
        intervals.append((start_idx, len(annot_array)))
        clusters.append(current_label)
        
    return intervals, clusters


def generate_color_map(unique_clusters):
    """Generates a dynamic color palette that scales based on the number of unique clusters."""
    cmap = plt.get_cmap('viridis', len(unique_clusters))
    colors = {}
    color_index = 0
    
    # Sorting works smoothly here because all elements are strings
    for cluster_id in sorted(unique_clusters, key=int):
        # sort by integer values to match the inference clusters
        colors[cluster_id] = cmap(color_index)
        color_index += 1
        
    return colors


def main():
    # parse arguments
    parser = argparse.ArgumentParser(description="Plot annotated spectrograms from multiple dictionaries.")
    parser.add_argument(
        "--dicts", 
        type=str, 
        nargs='+', 
        required=True, 
        help="Paths to the pickled annotation dictionaries (e.g., expanded_pkl_annotations.pkl processed_xml_annotations.pkl)"
    )
    parser.add_argument(
        "--wav_dir",
        type=str,
        required=True,
        help="Directory containing .wav files"
    )
    parser.add_argument(
        "--hop", 
        type=int, 
        required=True, 
        help="Sample hop length"
    )
    parser.add_argument(
        "--out_dir", 
        type=str, 
        default=".", 
        help="Directory to save outputs"
    )

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # loop through each dictionary provided as arguments
    for dict_path in args.dicts:
        print(f"\n--- Processing {dict_path} ---")
        
        # Load the dictionary where keys are filenames and values are the annotations
        with open(dict_path, "rb") as f:
            annot_dict = pickle.load(f)
            
        # Find all globally unique clusters to build a consistent colormap
        all_clusters = set()
        for annots in annot_dict.values():
            for val in annots:
                converted_val = convert_label(val)
                if converted_val != "-1":
                    all_clusters.add(converted_val)
        
        colors = generate_color_map(all_clusters)
        
        # Create a specific output subdirectory for this dictionary's plots
        dict_name = os.path.basename(dict_path).replace('.pkl', '')
        plot_out_dir = os.path.join(args.out_dir, dict_name)
        os.makedirs(plot_out_dir, exist_ok=True)
        
        # Iterate through dictionary and plot
        for wave_file, time_bins in annot_dict.items():
            wav_path = os.path.join(args.wav_dir, wave_file)
            
            if not os.path.exists(wav_path):
                print(f"Skipping {wave_file}, not found in {args.wav_dir}.")
                continue
                
            # Get intervals and corresponding cluster IDs for the sequence
            intervals, clusters = extract_intervals_and_clusters(time_bins)
            
            if not intervals:
                print(f"No annotations found for {wave_file}. Skipping plot.")
                continue
                
            # Load spectrogram of sample
            spectrogram, t_sec, samplerate = load_stft(wav_path, args.hop)
            
            save_name = os.path.join(plot_out_dir, f"annotated_spec_{wave_file.replace('.wav', '.png')}")
            print(f"Plotting {wave_file} with {len(intervals)} syllables...")

            # plot the spectrogram of each annotated wav file
            plot_spectrograms(
                spectrogram, 
                intervals, 
                t_sec, 
                samplerate,
                save_name, 
                colors, 
                clusters,
                args.hop
            )

if __name__ == "__main__":
    main()
