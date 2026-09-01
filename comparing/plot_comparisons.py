import os
import sys
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import re
import librosa
import soundfile as sf
from scipy.signal import ellip, filtfilt
import matplotlib.gridspec as gridspec

# Plot Markov Chain Transition Graphs
def draw_markov_chain(matrix, ax, title, pos, all_nodes, min_prob_threshold=0.05):
    """Utility function to render a transition matrix as a Markov Chain network diagram."""
    if matrix.empty:
        ax.set_title(f"{title}\n(No Data)", fontsize=11)
        ax.axis('off')
        return

    # Container graph representing positions
    G_dummy = nx.DiGraph()
    G_dummy.add_nodes_from(all_nodes)
    
    # Determine active vs inactive nodes for THIS matrix
    active_nodes = set(matrix.index)
    inactive_nodes = set(all_nodes) - active_nodes


    # Draw active nodes solid
    if active_nodes:
        nx.draw_networkx_nodes(G_dummy, pos, nodelist=list(active_nodes), ax=ax, 
                               node_color='#89CFF0', node_size=1000, edgecolors='black', alpha=1.0)
        nx.draw_networkx_labels(G_dummy, pos, labels={n: n for n in active_nodes}, ax=ax, 
                                font_size=9, font_weight='bold')
    # Draw inactive nodes translucent
    if inactive_nodes:
        nx.draw_networkx_nodes(G_dummy, pos, nodelist=list(inactive_nodes), ax=ax, 
                               node_color='#89CFF0', node_size=1000, edgecolors='black', alpha=0.15)
        nx.draw_networkx_labels(G_dummy, pos, labels={n: n for n in inactive_nodes}, ax=ax, 
                                font_size=9, font_weight='bold', alpha=0.0)

    # Helper function to plot distinct edge groups cleanly
    def draw_edges(edge_list):
        if not edge_list: return
        alpha = 1.0
        
        regular_edges = []
        self_loops = []
        weights_reg = []
        weights_self = []
        
        for src, dst, prob in edge_list:
            weight_scaled = (prob * 4.0)
            
            if src == dst:
                self_loops.append((src, dst))
                weights_self.append(weight_scaled)
            else:
                regular_edges.append((src, dst))
                weights_reg.append(weight_scaled)

        # Non self-loops
        if regular_edges:
            nx.draw_networkx_edges(
                G_dummy, pos, ax=ax, edgelist=regular_edges, width=weights_reg, 
                arrowstyle='->', arrowsize= 26, edge_color='#555555', 
                alpha=alpha, connectionstyle='arc3,rad=0.15'
            )
        
        # Self-loops (Using a larger dummy node_size forces networkx to draw broader self-loops)
        if self_loops:
            nx.draw_networkx_edges(
                G_dummy, pos, ax=ax, edgelist=self_loops, width=weights_self,
                arrowstyle='->', arrowsize= 26, edge_color='#555555', 
                alpha=alpha, node_size=2000 
            )

    # Draw active edges for the current matrix
    active_edges = []
    
    for src in active_nodes:
        for dst in active_nodes:
            prob = matrix.loc[src, dst]
            if prob >= min_prob_threshold:
                active_edges.append((src, dst, prob))
                
    draw_edges(active_edges)

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axis('off')

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

def annotate(ax, spec, extent, labels, sec_per_label, colors=None):
    """
    Plots a spectrogram on a given Matplotlib axis and adds colored
    highlight bars (annotations) just above the plot to indicate specific time intervals.
    """
    # Plot the 2D spectrogram array
    im = ax.imshow(spec, origin='lower', cmap='magma', aspect='auto', extent=extent)
    
    plot_start, plot_end = extent[0], extent[1]
    total_duration = plot_end - plot_start

    # Crop 25% off the beginning and 25% off the end of the entire spectrogram
    crop_start = plot_start + (0.25 * total_duration)
    crop_end = plot_end - (0.25 * total_duration)
    ax.set_xlim(crop_start, crop_end)

    # Convert crop boundaries into frame index bounds
    min_frame = int(0.25 * len(labels))
    max_frame = int(0.75 * len(labels))

    current_label = None
    run_start_frame = min_frame

    # Iterate strictly within the cropped frame window (+1 to flush the final segment)
    for i in range(min_frame, max_frame + 1):
        # Assign None at the boundary to force drawing the last open interval
        label = labels[i] if i < max_frame else None

        # Check for label transition
        if label != current_label:
            # Draw the previous valid interval (ignoring silence: -1 and None)
            if current_label is not None and current_label not in ["-1", -1, "-1.0"]:
                # Convert frame indices to time in seconds
                t_start = plot_start + (run_start_frame * sec_per_label)
                t_end = plot_start + (i * sec_per_label)

                # Fetch color 
                color = colors.get(current_label)

                # Draw horizontal colored bar
                ax.axvspan(
                    t_start, t_end,
                    ymin=1.02, ymax=1.08,
                    alpha=0.7, clip_on=False,
                    edgecolor='black', facecolor=color
                )

            # Reset start marker for the new label
            current_label = label
            run_start_frame = i

    ax.set_ylabel('Frequency Bin', fontsize=24)
    return im

def plot_spectrograms(label_dictionary, bird):
    """
    Creates a wide-format figure showing a spectrogram with annotated time intervals,
    attaches a colorbar, and saves the final plot to disk.
    """

    for k, v in label_dictionary.items():
        basename = os.path.basename(k)
        wav_number = int(os.path.splitext(basename)[0])

        if (wav_number % 10) != 0:
            continue

        file = os.path.join(f"3470165/{bird}/Wave/",k)
        spectrogram, t_sec, sample_rate = load_stft(file, 86)

        sec_per_label = 1.0 / sample_rate
        
        # Define the bounding box for the image data [x_min, x_max, y_min, y_max]
        img_extent = [0, t_sec, 0, spectrogram.shape[0]]

        # Initialize a large, wide figure (20x12)
        fig, ax = plt.subplots(1, 1, figsize=(20, 12), sharex=True)
        
        # Force the physical aspect ratio of the axes to be very wide and short (5:1 width-to-height)
        ax.set_box_aspect(0.2) 

        labels = v

        clusters = set(labels) - set([-1]) - set(["-1"]) - set(["-1.0"])   

        cmap = plt.get_cmap('viridis', len(set(clusters))) 
        colors = {}
        color_index = 0
    
        # Iterate through unique cluster IDs (sorted to ensure consistent color assignment)
        for cluster_id in sorted(set(clusters)): 
            # Assign a specific RGBA color tuple to each cluster ID
            colors[cluster_id] = cmap(color_index)
            color_index += 1

        # Call the helper function to draw the spectrogram and the top-edge annotations
        im = annotate(ax, spectrogram, img_extent, labels, sec_per_label, colors=colors)
        
        # Add a colorbar mapped to the spectrogram's intensity (dB)
        cbar = plt.colorbar(im, ax=ax, label='Intensity (dB)', shrink=0.22, aspect=10, pad=0.02)
        cbar.set_label('Intensity (dB)', fontsize=24)
        cbar.ax.tick_params(labelsize=20)
        
        ax.set_xlabel('Time (s)', fontsize=24)
        ax.set_ylabel("Frequency Bin", fontsize=24)
        ax.tick_params(axis='both', which='major', labelsize=20)
        
        # Adjust layout so labels/colorbars aren't cut off during saving
        plt.tight_layout()

        save_name = os.path.join(OUT_DIR, f"gt_spectrogram_{k}_{bird}.png")
        
        # Save the figure to the provided filepath (e.g., .png or .pdf), keeping all edges tight
        plt.savefig(save_name, bbox_inches='tight')
        
        plt.close()


STATS_DIR = sys.argv[1]  # Directory containing comparison_stats_{bird}.pkl files
OUT_DIR = sys.argv[1]      # Directory where generated figures will be saved

os.makedirs(OUT_DIR, exist_ok=True)

# Find all saved comparison stats PKL files
stats_files = sorted(glob.glob(os.path.join(STATS_DIR, "comparison_stats_*.pkl")))
gt_label_files = glob.glob(os.path.join(STATS_DIR, "processed_xml_annotations_*.pkl"))

 # Fallback to current working directory if not found in STATS_DIR
if not stats_files:
    stats_files = sorted(glob.glob("comparison_stats_*.pkl"))
if not gt_label_files:
    gt_label_files = glob.glob("processed_xml_annotations_*.pkl")

for label_file in gt_label_files:
    bird = os.path.basename(label_file).replace("processed_xml_annotations_", "").replace(".pkl", "")
    # open labels and plot spectrograms
    with open(label_file, 'rb') as file:
        data = pickle.load(file)
    plot_spectrograms(data, bird)

fer_records = []
similarity_records = []
trans_diff_records = []
bird_stats = {}

# Dictionaries to store counts for plot labels
recs_counts = {}
gt_counts = {}
pred_counts_tweety = {}
pred_counts_birdsong = {}

print(f"Found {len(stats_files)} bird stats files to process...")

# load and Format Data
for pkl_file in stats_files:
    # Extract bird name from filename
    filename = os.path.basename(pkl_file)
    bird_id = filename.replace("comparison_stats_", "").replace(".pkl", "")
    bird_id = re.sub(r"([a-zA-Z]+)(\d+)", r"\1 \2", bird_id)
    
    with open(pkl_file, "rb") as f:
        stats = pickle.load(f)
        bird_stats[bird_id] = stats
        
    # get transition probability matrices for plotting later
    gt_mat = stats.get('transition_matrix_gt', pd.DataFrame())
    tweety_mat = stats.get('transition_matrix_tweety', pd.DataFrame())
    birdsong_mat = stats.get('transition_matrix_birdsong', pd.DataFrame())

    print("Bird ID:", bird_id)
    print(gt_mat.index)
    print(tweety_mat.index)
    print(birdsong_mat.index)

    # Extract counts for labeling based on matrix row counts
    recs_counts[bird_id] = len(stats.get('per_record_fer_tweety', []))
    gt_counts[bird_id] = len(gt_mat) if not gt_mat.empty else 0
    pred_counts_tweety[bird_id] = len(tweety_mat) if not tweety_mat.empty else 0
    pred_counts_birdsong[bird_id] = len(birdsong_mat) if not birdsong_mat.empty else 0

    # Extract Frame Error Rates
    for fer in stats.get('per_record_fer_tweety', []):
        fer_records.append({'Bird': bird_id, 'Model': 'TweetyBERT', 'FER': fer})
    for fer in stats.get('per_record_fer_birdsong', []):
        fer_records.append({'Bird': bird_id, 'Model': 'Birdsong', 'FER': fer})

    # Extract similarity values
    for sim in stats.get('per_syb_similarity_tweety', []):
        similarity_records.append({'Bird': bird_id, 'Model': 'TweetyBERT', 'Similarity': sim})
    for sim in stats.get('per_syb_similarity_birdsong', []):
        similarity_records.append({'Bird': bird_id, 'Model': 'Birdsong', 'Similarity': sim})
        
    if not gt_mat.empty:
        # Calculate raw differences (Predicted - GT)
        tweety_diffs = (tweety_mat - gt_mat).values.flatten()
        birdsong_diffs = (birdsong_mat - gt_mat).values.flatten()
        
        for diff in tweety_diffs:
            trans_diff_records.append({'Bird': bird_id, 'Model': 'TweetyBERT', 'Difference': diff})
        for diff in birdsong_diffs:
            trans_diff_records.append({'Bird': bird_id, 'Model': 'Birdsong', 'Difference': diff})

df_fer = pd.DataFrame(fer_records)
df_similarity = pd.DataFrame(similarity_records)
df_trans_diff = pd.DataFrame(trans_diff_records)

# Color Palette
palette = {'TweetyBERT': 'red', 'Birdsong': 'blue'}

# Dynamically sort bird IDs by their numeric value
if not df_fer.empty:
    bird_order = sorted(
        df_fer['Bird'].unique(), 
        key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0
    )
else:
    bird_order = [f"Bird{i}" for i in range(11)]

# Plot Frame Error Rate Violins
if not df_fer.empty:
    plt.figure(figsize=(10, 5))
    ax = sns.violinplot(
        data=df_fer, x='Bird', y='FER', hue='Model', 
        split=True, inner=None, density_norm="width", order=bird_order, palette=palette, legend=False
    )
    sns.stripplot(
        data=df_fer, x='Bird', y='FER', hue='Model', 
        order=bird_order, palette=palette, dodge=True, alpha=0.8, size=4, 
        legend=False, linewidth=0.5, edgecolor='black'        
    )
    
    # Add count annotations inside the plot
    for i, b in enumerate(bird_order):
        annotation = f"N={recs_counts.get(b, 0)}\nGT={gt_counts.get(b, 0)}"
        ax.text(i, 0.75, annotation, transform=ax.get_xaxis_transform(),
                ha='center', va='bottom', fontsize=12, color='black')
    
    plt.title('Distribution of Frame Error Rate (FER) Per Recording Across Birds', fontsize=20, fontweight='bold')
    plt.ylabel('Frame Error Rate', fontsize=17)
    plt.ylim(-0.05, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    fer_plot_path = os.path.join(OUT_DIR, "violin_fer_per_bird.png")
    plt.savefig(fer_plot_path, dpi=600)
    plt.close()

# Plot similarity Index Violins
if not df_similarity.empty:
    plt.figure(figsize=(10, 6))
    ax = sns.violinplot(
        data=df_similarity, x='Bird', y='Similarity', hue='Model', 
        split=True, inner=None, order=bird_order, palette=palette, density_norm='width', cut=0, legend=False
    )
    sns.stripplot(
        data=df_similarity, x='Bird', y='Similarity', hue='Model', 
        order=bird_order, palette=palette, dodge=True, alpha=0.8, size=4, 
        legend=False, linewidth=0.5, edgecolor='black'          
    )
    
    # Add count annotations inside the plot
    for i, b in enumerate(bird_order):
        annotation = f"GT={gt_counts.get(b, 0)}\nTweety={pred_counts_tweety.get(b, 0)}\nCHURP={pred_counts_birdsong.get(b, 0)}"
        ax.text(i, 0.02, annotation, transform=ax.get_xaxis_transform(),
                ha='center', va='bottom', fontsize=12, color='black',
                bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', boxstyle='round,pad=0.3'))
    
    plt.title('Distributions of Min-Max Similarity Values Per Syllable Across Birds', fontsize=20, fontweight='bold')
    plt.ylabel('Min-Max Similarity', fontsize=17)
    plt.ylim(-0.05, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    similarity_plot_path = os.path.join(OUT_DIR, "violin_similarity_per_bird.png")
    plt.savefig(similarity_plot_path, dpi=600)
    plt.close()


# Render a figure for each bird containing GT, TweetyBERT, and birdsong Markov chains side-by-side
for bird_id, stats in bird_stats.items():
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))
    
    gt_mat = stats.get('transition_matrix_gt', pd.DataFrame())
    tweety_mat = stats.get('transition_matrix_tweety', pd.DataFrame())
    birdsong_mat = stats.get('transition_matrix_birdsong', pd.DataFrame())
    
    for mat in [gt_mat, tweety_mat, birdsong_mat]:
        if not mat.empty:
            mat.index = mat.index.astype(str)
            mat.columns = mat.columns.astype(str)

    # Establish ground truth nodes to fix coordinates/layout across all 3 graphs
    all_nodes = set()
    all_nodes = sorted(list(gt_mat.index))
    
    # Reindex all matrices to align perfectly
    if not gt_mat.empty: gt_mat_filled = gt_mat.reindex(index=all_nodes, columns=all_nodes, fill_value=0.0)
    if not tweety_mat.empty: tweety_mat_filled = tweety_mat.reindex(index=all_nodes, columns=all_nodes, fill_value=0.0)
    if not birdsong_mat.empty: birdsong_mat_filled = birdsong_mat.reindex(index=all_nodes, columns=all_nodes, fill_value=0.0)

    # Ground truth circular layout
    pos = nx.circular_layout(all_nodes)
    
    # Calculate Manhattan Distances for the title using aligned matrices
    if not gt_mat_filled.empty:
        tweety_manhattan = (gt_mat_filled - tweety_mat_filled).abs().sum().sum()
        birdsong_manhattan = (gt_mat_filled - birdsong_mat_filled).abs().sum().sum()        
        tweety_title = f"TweetyBERT\n(Manhattan Dist: {tweety_manhattan:.2f})"
        birdsong_title = f"CHURP\n(Manhattan Dist: {birdsong_manhattan:.2f})"
    else:
        tweety_title = "TweetyBERT"
        birdsong_title = "CHURP"

    # provide unfilled matrices to the draw_markov_chain function to ensure proper handling of empty matrices
    draw_markov_chain(gt_mat, axes[0], "Ground Truth", pos, all_nodes)
    draw_markov_chain(tweety_mat, axes[1], tweety_title, pos, all_nodes)
    draw_markov_chain(birdsong_mat, axes[2], birdsong_title, pos, all_nodes)
    
    plt.suptitle(f"Syllable Transition Markov Chains — {bird_id}", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    mc_plot_path = os.path.join(OUT_DIR, f"markov_chain_{bird_id}.png")
    plt.savefig(mc_plot_path, dpi=300)
    plt.close()
    print(f"Saved Markov Chain Diagram for {bird_id} -> {mc_plot_path}")

print(f"\nAll plots successfully saved to directory: '{OUT_DIR}/'")
