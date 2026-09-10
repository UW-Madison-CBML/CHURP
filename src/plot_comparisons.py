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

def draw_markov_chain(matrix, ax, title, pos, all_nodes, min_prob_threshold=0.05):
    """Renders a transition matrix as a Markov Chain network diagram."""
    
    # Handle edge case where the transition matrix contains no data
    if matrix.empty:
        ax.set_title(f"{title}\n(No Data)", fontsize=22.5)
        ax.axis('off')
        return

    # Initialize a directed graph to serve as a spatial container for node positions
    G_dummy = nx.DiGraph()
    G_dummy.add_nodes_from(all_nodes)
    
    # Identify which nodes are present in the current matrix versus the global set
    active_nodes = set(matrix.index)
    inactive_nodes = set(all_nodes) - active_nodes

    # Render active nodes with solid colors and bold labels
    if active_nodes:
        nx.draw_networkx_nodes(G_dummy, pos, nodelist=list(active_nodes), ax=ax, 
                               node_color='#89CFF0', node_size=1500, edgecolors='black', alpha=1.0)
        nx.draw_networkx_labels(G_dummy, pos, labels={n: n for n in active_nodes}, ax=ax, 
                                font_size=18, font_weight='bold')
                                
    # Render inactive nodes with high transparency
    if inactive_nodes:
        nx.draw_networkx_nodes(G_dummy, pos, nodelist=list(inactive_nodes), ax=ax, 
                               node_color='#89CFF0', node_size=1500, edgecolors='black', alpha=0.15)
        nx.draw_networkx_labels(G_dummy, pos, labels={n: n for n in inactive_nodes}, ax=ax, 
                                font_size=13.5, font_weight='bold', alpha=0.0)

    # Define a nested helper to separate and style self-loops versus standard directed edges
    def draw_edges(edge_list):
        if not edge_list: return
        alpha = 1.0
        
        regular_edges = []
        self_loops = []
        weights_reg = []
        weights_self = []
        
        # Scale edge thickness based on transition probability for visual weight
        for src, dst, prob in edge_list:
            weight_scaled = (prob * 6.0) 
            
            if src == dst:
                self_loops.append((src, dst))
                weights_self.append(weight_scaled)
            else:
                regular_edges.append((src, dst))
                weights_reg.append(weight_scaled)

        # Plot standard node-to-node transitions with curved arrows
        if regular_edges:
            nx.draw_networkx_edges(
                G_dummy, pos, ax=ax, edgelist=regular_edges, width=weights_reg, 
                arrowstyle='->', arrowsize=39, edge_color='#555555', 
                alpha=alpha, connectionstyle='arc3,rad=0.15'
            )
        
        # Plot self-transitions (node to itself) with enlarged node boundaries to prevent overlap
        if self_loops:
            nx.draw_networkx_edges(
                G_dummy, pos, ax=ax, edgelist=self_loops, width=weights_self,
                arrowstyle='->', arrowsize=39, edge_color='#555555', 
                alpha=alpha, node_size=3000  # Scaled up from 2000
            )

    # Filter and collect edges that meet the minimum probability threshold for visualization
    active_edges = []
    
    for src in active_nodes:
        for dst in active_nodes:
            prob = matrix.loc[src, dst]
            if prob >= min_prob_threshold:
                active_edges.append((src, dst, prob))
                
    # Execute the edge drawing helper and apply final axis formatting            
    draw_edges(active_edges)

    ax.set_title(title, fontsize=24, fontweight='bold', pad=30)
    ax.axis('off')

def load_stft(f, hop_length):
    """
    Helper for loading individual wav files as spectrograms for plotting purposes
    """
    # Load the target audio file as a single-channel integer array
    with sf.SoundFile(f, 'r') as wav_file:
        samplerate = wav_file.samplerate
        total_frames = wav_file.frames
        data = wav_file.read(dtype='int16')

    # Apply a 500Hz high-pass elliptic filter to remove low-frequency noise (applied forwards and backwards for zero phase)
    b, a = ellip(5, 0.2, 40, 500 / (samplerate / 2), 'high')
    data = filtfilt(b, a, data)

    # Compute the Short-Time Fourier Transform (STFT) and convert amplitude magnitude to a decibel scale
    Sxx = librosa.stft(data.astype(float), n_fft=1024, hop_length=hop_length, window='hann')
    Sxx_log = librosa.amplitude_to_db(np.abs(Sxx), ref=np.max)
    
    # Calculate the total audio duration in seconds
    t_sec = total_frames / samplerate

    return Sxx_log, t_sec, samplerate

def annotate(ax, spec, extent, labels, sec_per_label, colors=None):
    """
    Plots a spectrogram on a given Matplotlib axis and adds colored
    highlight bars (annotations) just above the plot to indicate specific time intervals.
    """
    # Render the 2D spectrogram array on the provided matplotlib axis
    im = ax.imshow(spec, origin='lower', cmap='magma', aspect='auto', extent=extent)
    
    plot_start, plot_end = extent[0], extent[1]
    total_duration = plot_end - plot_start

    # Calculate bounds to crop 25% of the visual space from both the start and end of the spectrogram
    crop_start = plot_start + (0.25 * total_duration)
    crop_end = plot_end - (0.25 * total_duration)
    ax.set_xlim(crop_start, crop_end)

    # Convert the cropped time boundaries into corresponding frame indices
    min_frame = int(0.25 * len(labels))
    max_frame = int(0.75 * len(labels))

    current_label = None
    run_start_frame = min_frame

    # Iterate strictly within the cropped frame window (+1 to flush the final segment)
    for i in range(min_frame, max_frame + 1):
        # Assign None at the boundary to force drawing the last open interval
        label = labels[i] if i < max_frame else None

        # Detect transitions between different cluster labels to draw bounding boxes
        if label != current_label:
            # Draw the previous valid interval (ignoring silence/background tags like -1)
            if current_label is not None and current_label not in ["-1", -1, "-1.0"]:
                # Convert frame indices to time in seconds
                t_start = plot_start + (run_start_frame * sec_per_label)
                t_end = plot_start + (i * sec_per_label)

                # Fetch color mapped to the current syllable cluster
                color = colors.get(current_label)

                # Draw a horizontal colored bar above the spectrogram corresponding to the specific cluster
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
    # Iterate through the dictionary of audio files and their corresponding frame labels
    for k, v in label_dictionary.items():
        basename = os.path.basename(k)
        wav_number = int(os.path.splitext(basename)[0])

        # Process only every 50th file to reduce visualization overhead
        if (wav_number % 50) != 0:
            continue

        # Construct the filepath and load the processed spectrogram
        file = os.path.join(f"3470165/{bird}/Wave/",k)
        spectrogram, t_sec, sample_rate = load_stft(file, 86)

        sec_per_label = 1.0 / sample_rate
        
        # Define the bounding box for the image data [x_min, x_max, y_min, y_max]
        img_extent = [0, t_sec, 0, spectrogram.shape[0]]

        # Initialize a wide-format figure (20x12) and force a very wide, short aspect ratio (5:1)
        fig, ax = plt.subplots(1, 1, figsize=(20, 12), sharex=True)
        ax.set_box_aspect(0.2) 

        labels = v

        # Identify unique valid cluster IDs, filtering out all variations of background/silence labels
        clusters = set(labels) - set([-1]) - set(["-1"]) - set(["-1.0"])   

        # Generate a distinct color palette based on the number of unique clusters
        cmap = plt.get_cmap('viridis', len(set(clusters))) 
        colors = {}
        color_index = 0
    
        # Assign a specific RGBA color tuple to each cluster ID (sorted to ensure consistent color assignment)
        for cluster_id in sorted(set(clusters)): 
            colors[cluster_id] = cmap(color_index)
            color_index += 1

        # Delegate to the helper function to draw the spectrogram and its top-edge label annotations
        im = annotate(ax, spectrogram, img_extent, labels, sec_per_label, colors=colors)
        
        # Attach and format a colorbar mapped to the spectrogram's decibel intensity
        cbar = plt.colorbar(im, ax=ax, label='Intensity (dB)', shrink=0.22, aspect=10, pad=0.02)
        cbar.set_label('Intensity (dB)', fontsize=24)
        cbar.ax.tick_params(labelsize=20)
        
        ax.set_xlabel('Time (s)', fontsize=24)
        ax.set_ylabel("Frequency Bin", fontsize=24)
        ax.tick_params(axis='both', which='major', labelsize=20)
        
        # Save the figure to disk, ensuring labels and colorbars are not cropped during tight layout
        plt.tight_layout()

        save_name = os.path.join(OUT_DIR, f"gt_spectrogram_{k}_{bird}.png")
        plt.savefig(save_name, bbox_inches='tight')
        
        plt.close()

# Parse input arguments for input and output directories
STATS_DIR = sys.argv[1]  # Directory containing comparison_stats_{bird}.pkl files
OUT_DIR = sys.argv[1]      # Directory where generated figures will be saved (same as STATS_DIR)

# Ensure the output directory exists before generating figures
os.makedirs(OUT_DIR, exist_ok=True)

# Locate all saved comparison statistics and ground truth label Pickle files
stats_files = sorted(glob.glob(os.path.join(STATS_DIR, "comparison_stats_*.pkl")))
gt_label_files = glob.glob(os.path.join(STATS_DIR, "processed_xml_annotations_*.pkl"))

# Provide a fallback to the current working directory if files aren't found in the specified path
if not stats_files:
    stats_files = sorted(glob.glob("comparison_stats_*.pkl"))
if not gt_label_files:
    gt_label_files = glob.glob("processed_xml_annotations_*.pkl")

# Parse bird identifiers from filenames and trigger spectrogram generation for each
for label_file in gt_label_files:
    bird = os.path.basename(label_file).replace("processed_xml_annotations_", "").replace(".pkl", "")
    with open(label_file, 'rb') as file:
        data = pickle.load(file)
    plot_spectrograms(data, bird)

# Initialize data structures to collect metrics for downstream statistical plotting
fer_records = []
similarity_records = []
trans_diff_records = []
bird_stats = {}

# Dictionaries to store structural counts for plot annotations
recs_counts = {}
gt_counts = {}
pred_counts_tweety = {}
pred_counts_birdsong = {}

print(f"Found {len(stats_files)} bird stats files to process...")

# Extract and Format Data Iteratively
for pkl_file in stats_files:
    # Extract and format the bird ID from the filename (e.g., 'bird1' -> 'bird 1')
    filename = os.path.basename(pkl_file)
    bird_id = filename.replace("comparison_stats_", "").replace(".pkl", "")
    bird_id = re.sub(r"([a-zA-Z]+)(\d+)", r"\1 \2", bird_id)
    
    # Load the dictionary of transition statistics for the current bird
    with open(pkl_file, "rb") as f:
        stats = pickle.load(f)
        bird_stats[bird_id] = stats
        
    # Extract transition probability matrices for ground truth and both models
    gt_mat = stats.get('transition_matrix_gt', pd.DataFrame())
    tweety_mat = stats.get('transition_matrix_tweety', pd.DataFrame())
    birdsong_mat = stats.get('transition_matrix_birdsong', pd.DataFrame())

    print("Bird ID:", bird_id)
    print(gt_mat.index)
    print(tweety_mat.index)
    print(birdsong_mat.index)

    # Record structural counts (records, unique syllables) for plot annotations
    recs_counts[bird_id] = len(stats.get('per_record_fer_tweety', []))
    gt_counts[bird_id] = len(gt_mat) if not gt_mat.empty else 0
    pred_counts_tweety[bird_id] = len(tweety_mat) if not tweety_mat.empty else 0
    pred_counts_birdsong[bird_id] = len(birdsong_mat) if not birdsong_mat.empty else 0

    # Unpack Frame Error Rates (FER) into flat records for seaborn plotting
    for fer in stats.get('per_record_fer_tweety', []):
        fer_records.append({'Bird': bird_id, 'Model': 'TweetyBERT', 'FER': fer})
    for fer in stats.get('per_record_fer_birdsong', []):
        fer_records.append({'Bird': bird_id, 'Model': 'Birdsong', 'FER': fer})

    # Unpack similarity indices into flat records for seaborn plotting
    for sim in stats.get('per_syb_similarity_tweety', []):
        similarity_records.append({'Bird': bird_id, 'Model': 'TweetyBERT', 'Similarity': sim})
    for sim in stats.get('per_syb_similarity_birdsong', []):
        similarity_records.append({'Bird': bird_id, 'Model': 'Birdsong', 'Similarity': sim})
        
    # If ground truth exists, calculate and record the raw differences in transition probabilities
    if not gt_mat.empty:
        tweety_diffs = (tweety_mat - gt_mat).values.flatten()
        birdsong_diffs = (birdsong_mat - gt_mat).values.flatten()
        
        for diff in tweety_diffs:
            trans_diff_records.append({'Bird': bird_id, 'Model': 'TweetyBERT', 'Difference': diff})
        for diff in birdsong_diffs:
            trans_diff_records.append({'Bird': bird_id, 'Model': 'Birdsong', 'Difference': diff})

# Convert flat metric records into Pandas DataFrames for easier visualization
df_fer = pd.DataFrame(fer_records)
df_similarity = pd.DataFrame(similarity_records)
df_trans_diff = pd.DataFrame(trans_diff_records)

# Define standard color mappings for consistency across models
palette = {'TweetyBERT': 'red', 'Birdsong': 'blue'}

# Dynamically sort bird IDs numerically to ensure logical ordering on plot axes
if not df_fer.empty:
    bird_order = sorted(
        df_fer['Bird'].unique(), 
        key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0
    )
else:
    bird_order = [f"Bird{i}" for i in range(11)]

# Generate a split violin plot overlayed with a stripplot to show FER distributions
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
    
    # Embed specific sample size and ground truth counts directly inside the plot
    for i, b in enumerate(bird_order):
        annotation = f"N={recs_counts.get(b, 0)}\nGT={gt_counts.get(b, 0)}"
        ax.text(i, 0.75, annotation, transform=ax.get_xaxis_transform(),
                ha='center', va='bottom', fontsize=12, color='black')
    
    # Format plot titles, labels, and axes limits before saving 
    plt.title('Distributions of Frame Error Rate Per Recording Across Birds', fontsize=19, fontweight='bold')
    plt.ylabel('Frame Error Rate', fontsize=17)
    plt.xlabel('')
    plt.ylim(-0.05, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    fer_plot_path = os.path.join(OUT_DIR, "violin_fer_per_bird.png")
    plt.savefig(fer_plot_path, dpi=600)
    plt.close()

# Generate a split violin plot overlayed with a stripplot to show Ruzicka Similarity distributions
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
    
    # Embed specific syllable counts for Ground Truth, Tweety, and CHURP directly inside the plot
    for i, b in enumerate(bird_order):
        annotation = f"GT={gt_counts.get(b, 0)}\nTweety={pred_counts_tweety.get(b, 0)}\nCHURP={pred_counts_birdsong.get(b, 0)}"
        ax.text(i, 0.02, annotation, transform=ax.get_xaxis_transform(),
                ha='center', va='bottom', fontsize=10, color='black',
                bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', boxstyle='round,pad=0.3'))
    
    # Format plot titles, labels, and axes limits before saving
    plt.title('Distributions of Ruzicka Similarity Values Per Syllable Across Birds', fontsize=17, fontweight='bold')
    plt.ylabel('Ruzicka Similarity', fontsize=17)
    plt.xlabel('')
    plt.ylim(-0.05, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    similarity_plot_path = os.path.join(OUT_DIR, "violin_similarity_per_bird.png")
    plt.savefig(similarity_plot_path, dpi=600)
    plt.close()


    # Generate an identical plot to the Ruzicka similarity one, but labeled specifically as Jaccard Index
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
    
    # Embed specific syllable counts directly inside the plot
    for i, b in enumerate(bird_order):
        annotation = f"GT={gt_counts.get(b, 0)}\nTweety={pred_counts_tweety.get(b, 0)}\nCHURP={pred_counts_birdsong.get(b, 0)}"
        ax.text(i, 0.02, annotation, transform=ax.get_xaxis_transform(),
                ha='center', va='bottom', fontsize=10, color='black',
                bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', boxstyle='round,pad=0.3'))

    # Format plot titles, labels, and axes limits before saving (Font size decreased by 1 (18 -> 17))
    plt.title('Distributions of Jaccard Index Values Per Syllable Across Birds', fontsize=17, fontweight='bold')
    plt.ylabel('Jaccard Index', fontsize=17)
    plt.xlabel('')
    plt.ylim(-0.05, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    similarity_plot_path = os.path.join(OUT_DIR, "violin_similarity_per_bird_jaccard.png")
    plt.savefig(similarity_plot_path, dpi=600)
    plt.close()


distance_records = []

# Iterate through collected bird statistics to render their Markov Chain comparisons in 3 side-by-side plots
for bird_id, stats in bird_stats.items():
    # Initialize a 1x3 subplot figure with an adjusted wide footprint
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    
    gt_mat = stats.get('transition_matrix_gt', pd.DataFrame())
    tweety_mat = stats.get('transition_matrix_tweety', pd.DataFrame())
    birdsong_mat = stats.get('transition_matrix_birdsong', pd.DataFrame())
    
    # Ensure all matrix row and column indices are cast to strings for alignment consistency
    for mat in [gt_mat, tweety_mat, birdsong_mat]:
        if not mat.empty:
            mat.index = mat.index.astype(str)
            mat.columns = mat.columns.astype(str)

    # Establish a master set of all ground truth nodes to lock coordinate layouts across all 3 graphs
    all_nodes = set()
    all_nodes = sorted(list(gt_mat.index))
    
    # Reindex all transition matrices so they align perfectly in shape and node order
    if not gt_mat.empty: gt_mat_filled = gt_mat.reindex(index=all_nodes, columns=all_nodes, fill_value=0.0)
    if not tweety_mat.empty: tweety_mat_filled = tweety_mat.reindex(index=all_nodes, columns=all_nodes, fill_value=0.0)
    if not birdsong_mat.empty: birdsong_mat_filled = birdsong_mat.reindex(index=all_nodes, columns=all_nodes, fill_value=0.0)

    # Compute a fixed circular spatial layout for all graphs based on the master node list
    pos = nx.circular_layout(all_nodes)
    
    # Calculate Manhattan and Euclidean Distances between the predictive models and the ground truth
    if not gt_mat_filled.empty:
        tweety_manhattan = (gt_mat_filled - tweety_mat_filled).abs().sum().sum()
        birdsong_manhattan = (gt_mat_filled - birdsong_mat_filled).abs().sum().sum()
        tweety_euclidean = ((gt_mat_filled - tweety_mat_filled) ** 2).sum().sum() ** 0.5
        birdsong_euclidean = ((gt_mat_filled - birdsong_mat_filled) ** 2).sum().sum() ** 0.5
        tweety_title = f"TweetyBERT\nManhattan: {tweety_manhattan:.2f} / Euclidean: {tweety_euclidean:.2f}"
        birdsong_title = f"CHURP\nManhattan: {birdsong_manhattan:.2f} / Euclidean: {birdsong_euclidean:.2f}"
        
        # Archive the calculated distances for the final summary table
        distance_records.append({
            'Bird': bird_id,
            'Tweety Manhattan': f"{tweety_manhattan:.2f}",
            'Tweety Euclidean': f"{tweety_euclidean:.2f}",
            'CHURP Manhattan': f"{birdsong_manhattan:.2f}",
            'CHURP Euclidean': f"{birdsong_euclidean:.2f}"
        })
    else:
        tweety_title = "TweetyBERT"
        birdsong_title = "CHURP"

    # Render the three Markov Chains (Ground Truth, TweetyBERT, CHURP), providing unfilled matrices to safely handle empty inputs
    draw_markov_chain(gt_mat, axes[0], "Ground Truth", pos, all_nodes)
    draw_markov_chain(tweety_mat, axes[1], tweety_title, pos, all_nodes)
    draw_markov_chain(birdsong_mat, axes[2], birdsong_title, pos, all_nodes)
    
    # Finalize subplot spacing, add master titles, and save the figure
    plt.suptitle(f"Syllable Transition Markov Chains — {bird_id}", fontsize=20, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    mc_plot_path = os.path.join(OUT_DIR, f"markov_chain_{bird_id}.png")
    plt.savefig(mc_plot_path, dpi=600)
    plt.close()
    print(f"Saved Markov Chain Diagram for {bird_id} -> {mc_plot_path}")

# Generate Table with Manhattan and Euclidean Distances
if distance_records:
    df_distances = pd.DataFrame(distance_records)
    
    # Apply identical numeric sorting to the table rows to match the ordering in the violin plots
    df_distances['SortKey'] = df_distances['Bird'].apply(
        lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0
    )
    df_distances = df_distances.sort_values('SortKey').drop(columns='SortKey')

    # Merge TweetyBERT and CHURP metrics into shared columns separated by newlines for visual compactness
    compact_df = pd.DataFrame({
        'Bird': df_distances['Bird'],
        'Euclidean Dist\nTweety/CHURP': df_distances['Tweety Euclidean'] + "/" + df_distances['CHURP Euclidean'],
        'Manhattan Dist\nTweety/CHURP': df_distances['Tweety Manhattan'] + "/" + df_distances['CHURP Manhattan']
    })

    # Instantiate an un-axised Matplotlib figure tailored tightly to the table's dimensions
    fig, ax = plt.subplots(figsize=(5.5, len(compact_df) * 0.25 + 0.5))
    ax.axis('tight')
    ax.axis('off')
    
    # Render the pandas DataFrame directly as a graphical Matplotlib table
    table = ax.table(cellText=compact_df.values, 
                     colLabels=compact_df.columns, 
                     cellLoc='center', 
                     loc='center')
    
    # Override default font sizing and row scaling to compress the table's visual footprint
    table.auto_set_font_size(False)
    table.set_fontsize(9)  # Smaller font
    table.scale(1, 1.2)    # Reduced row height scaling

    # Automatically shrink column widths to fit the text tightly
    table.auto_set_column_width(col=list(range(len(compact_df.columns))))
    
    # Double the height of the header row cells to prevent overlap with newline characters
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_height(cell.get_height() * 2)
    
    plt.title("Distance Metrics by Bird", fontsize=11, fontweight='bold', pad=10)
    
    # Minimize plot padding around the table
    plt.tight_layout(pad=0)
    
    table_path = os.path.join(OUT_DIR, "distance_metrics_table.png")
    
    # Export the final table graphic, using bbox_inches to crop out any remaining whitespace
    plt.savefig(table_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Saved Compact Distance Metrics Table -> {table_path}")

# Notify user of successful pipeline completion
print(f"\nAll plots successfully saved to directory: '{OUT_DIR}/'")
