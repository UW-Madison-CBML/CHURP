import os
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import re

# --- Configuration ---
STATS_DIR = "comparison_outputs"  # Directory containing comparison_stats_{bird}.pkl files
OUT_DIR = "comparison_plots"      # Directory where generated figures will be saved

os.makedirs(OUT_DIR, exist_ok=True)

# Find all saved comparison stats PKL files
stats_files = sorted(glob.glob(os.path.join(STATS_DIR, "comparison_stats_*.pkl")))

if not stats_files:
    # Fallback to current working directory if not found in STATS_DIR
    stats_files = sorted(glob.glob("comparison_stats_*.pkl"))

fer_records = []
similarity_records = []
trans_diff_records = []
bird_stats = {}

print(f"Found {len(stats_files)} bird stats files to process...")

# --- 1. Load and Format Data ---
for pkl_file in stats_files:
    # Extract bird name from filename (e.g., 'comparison_stats_Bird0.pkl' -> 'Bird0')
    filename = os.path.basename(pkl_file)
    bird_id = filename.replace("comparison_stats_", "").replace(".pkl", "")
    
    with open(pkl_file, "rb") as f:
        stats = pickle.load(f)
        bird_stats[bird_id] = stats

    # Extract Frame Error Rates (FER)
    for fer in stats.get('per_record_fer_tweety', []):
        fer_records.append({'Bird': bird_id, 'Model': 'TweetyBERT', 'FER': fer})
    for fer in stats.get('per_record_fer_birdsong', []):
        fer_records.append({'Bird': bird_id, 'Model': 'Birdsong', 'FER': fer})

    # Extract similarity values
    for sim in stats.get('per_syb_similarity_tweety', []):
        similarity_records.append({'Bird': bird_id, 'Model': 'TweetyBERT', 'Similarity': sim})
    for sim in stats.get('per_syb_similarity_birdsong', []):
        similarity_records.append({'Bird': bird_id, 'Model': 'Birdsong', 'Similarity': sim})
        
    # Compute transition probability differences for violin plots
    gt_mat = stats.get('transition_matrix_gt', pd.DataFrame())
    tweety_mat = stats.get('transition_matrix_tweety', pd.DataFrame())
    birdsong_mat = stats.get('transition_matrix_birdsong', pd.DataFrame())
    
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

# Dynamically sort bird IDs by their numeric value (e.g., 0, 1, 2 ... 10)
if not df_fer.empty:
    bird_order = sorted(
        df_fer['Bird'].unique(), 
        key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0
    )
else:
    bird_order = [f"Bird{i}" for i in range(11)] # Fallback if dataframe is empty

# --- 2. Plot Frame Error Rate (FER) Violins ---
if not df_fer.empty:
    plt.figure(figsize=(10, 6))
    sns.violinplot(
        data=df_fer, 
        x='Bird', 
        y='FER', 
        hue='Model', 
        split=True, 
        inner=None,
        order=bird_order,
        palette=palette,
        cut=0
    )
    plt.title('Distribution of Frame Error Rate (FER) Per Recording Across Birds', fontsize=14, fontweight='bold')
    plt.ylabel('Frame Error Rate', fontsize=12)
    plt.xlabel('Bird ID', fontsize=12)
    plt.ylim(-0.05, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(title='Model', loc='upper right')
    plt.tight_layout()
    
    fer_plot_path = os.path.join(OUT_DIR, "violin_fer_per_bird.png")
    plt.savefig(fer_plot_path, dpi=300)
    plt.close()
    print(f"Saved FER Violin Plot -> {fer_plot_path}")

# --- 3. Plot similarity Index Violins ---
if not df_similarity.empty:
    plt.figure(figsize=(10, 6))
    sns.violinplot(
        data=df_similarity, 
        x='Bird', 
        y='Similarity', 
        hue='Model', 
        split=True, 
        inner=None,
        order=bird_order,
        palette=palette,
        cut=0
    )
    plt.title('Distributions of Min-Max Similarity Values Per Syllable Across Birds', fontsize=14, fontweight='bold')
    plt.ylabel('Min-Max Similarity', fontsize=12)
    plt.xlabel('Bird ID', fontsize=12)
    plt.ylim(-0.05, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(title='Model', loc='upper right')
    plt.tight_layout()
    
    similarity_plot_path = os.path.join(OUT_DIR, "violin_similarity_per_bird.png")
    plt.savefig(similarity_plot_path, dpi=300)
    plt.close()
    print(f"Saved Similarity Violin Plot -> {similarity_plot_path}")

# --- 4. Plot Transition Probability Differences (Predicted - GT) ---
if not df_trans_diff.empty:
    plt.figure(figsize=(10, 6))
    sns.violinplot(
        data=df_trans_diff, 
        x='Bird', 
        y='Difference', 
        hue='Model', 
        split=True, 
        inner=None,
        order=bird_order,
        palette=palette,
        cut=0
    )
    plt.title('Differences in Transition Probabilities (Predicted - Ground Truth)', fontsize=14, fontweight='bold')
    plt.ylabel('Transition Probability Difference', fontsize=12)
    plt.xlabel('Bird ID', fontsize=12)
    plt.axhline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.7) # Zero reference line
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(title='Model', loc='upper right')
    plt.tight_layout()
    
    trans_diff_plot_path = os.path.join(OUT_DIR, "violin_trans_diff_per_bird.png")
    plt.savefig(trans_diff_plot_path, dpi=300)
    plt.close()
    print(f"Saved Transition Differences Violin Plot -> {trans_diff_plot_path}")

# --- 5. Plot Markov Chain Transition Graphs ---
def draw_markov_chain(matrix, ax, title, min_prob_threshold=0.05):
    """Utility function to render a transition matrix as a Markov Chain network diagram."""
    G = nx.DiGraph()
    
    print(matrix)

    if isinstance(matrix, pd.DataFrame):
        nodes = [str(col) for col in matrix.columns]
        mat_vals = matrix.values
    else:
        nodes = [str(i) for i in range(matrix.shape[0])]
        mat_vals = matrix

    for node in nodes:
        G.add_node(node)
        
    for i, src in enumerate(nodes):
        for j, dst in enumerate(nodes):
            prob = mat_vals[i, j]
            if prob >= min_prob_threshold:  # Only render edges exceeding threshold
                G.add_edge(src, dst, weight=prob)

    if len(G.nodes) == 0:
        ax.set_title(f"{title}\n(No Transitions)", fontsize=11)
        ax.axis('off')
        return

    # Use circular layout for clear state transition flow
    pos = nx.circular_layout(G)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='#89CFF0', node_size=1000, edgecolors='black')
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=9, font_weight='bold')
    
    # Scale edge thickness and arrows based on transition probability
    edges = G.edges(data=True)
    if edges:
        weights = [e[2]['weight'] * 4.0 for e in edges]
        nx.draw_networkx_edges(
            G, pos, ax=ax, 
            edgelist=[(u, v) for u, v, _ in edges],
            width=weights, 
            arrowstyle='->', 
            arrowsize=14, 
            edge_color='#555555', 
            connectionstyle='arc3,rad=0.15'
        )
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axis('off')
git clone git@github.com:username/user-private-repository.git
# Render a figure for each bird containing GT, TweetyBERT, and CHURP Markov chains side-by-side
for bird_id, stats in bird_stats.items():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    
    gt_mat = stats.get('transition_matrix_gt', pd.DataFrame())
    tweety_mat = stats.get('transition_matrix_tweety', pd.DataFrame())
    birdsong_mat = stats.get('transition_matrix_birdsong', pd.DataFrame())
    
    # Calculate Manhattan Distances for the title using aligned matrices
    if not gt_mat.empty:
        gt_t, tweety_t = gt_mat.align(tweety_mat, join='outer', fill_value=0)
        gt_b, birdsong_b = gt_mat.align(birdsong_mat, join='outer', fill_value=0)
        
        tweety_manhattan = (gt_t - tweety_t).abs().sum().sum()
        birdsong_manhattan = (gt_b - birdsong_b).abs().sum().sum()        
        tweety_title = f"TweetyBERT\n(Manhattan Dist: {tweety_manhattan:.2f})"
        birdsong_title = f"Birdsong\n(Manhattan Dist: {birdsong_manhattan:.2f})"
    else:
        tweety_title = "TweetyBERT"
        birdsong_title = "Birdsong"
    
    draw_markov_chain(gt_mat, axes[0], "Ground Truth")
    draw_markov_chain(tweety_mat, axes[1], tweety_title)
    draw_markov_chain(birdsong_mat, axes[2], birdsong_title)
    
    plt.suptitle(f"Syllable Transition Markov Chains — {bird_id}", fontsize=15, fontweight='bold')
    plt.tight_layout()
    
    mc_plot_path = os.path.join(OUT_DIR, f"markov_chain_{bird_id}.png")
    plt.savefig(mc_plot_path, dpi=300)
    plt.close()
    print(f"Saved Markov Chain Diagram for {bird_id} -> {mc_plot_path}")

print(f"\nAll plots successfully saved to directory: '{OUT_DIR}/'")