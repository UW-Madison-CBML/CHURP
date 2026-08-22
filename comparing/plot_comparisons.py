import os
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import re

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
    plt.figure(figsize=(10, 6))
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
                ha='center', va='bottom', fontsize=9, color='black')
    
    plt.title('Distribution of Frame Error Rate (FER) Per Recording Across Birds', fontsize=14, fontweight='bold')
    plt.ylabel('Frame Error Rate', fontsize=12)
    plt.xlabel('Bird ID', fontsize=12)
    plt.ylim(-0.05, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    fer_plot_path = os.path.join(OUT_DIR, "violin_fer_per_bird.png")
    plt.savefig(fer_plot_path, dpi=300)
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
        annotation = f"GT={gt_counts.get(b, 0)}\nPred T={pred_counts_tweety.get(b, 0)}\nPred B={pred_counts_birdsong.get(b, 0)}"
        ax.text(i, 0.02, annotation, transform=ax.get_xaxis_transform(),
                ha='center', va='bottom', fontsize=8, color='black',
                bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', boxstyle='round,pad=0.3'))
    
    plt.title('Distributions of Min-Max Similarity Values Per Syllable Across Birds', fontsize=14, fontweight='bold')
    plt.ylabel('Min-Max Similarity', fontsize=12)
    plt.xlabel('Bird ID', fontsize=12)
    plt.ylim(-0.05, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    similarity_plot_path = os.path.join(OUT_DIR, "violin_similarity_per_bird.png")
    plt.savefig(similarity_plot_path, dpi=300)
    plt.close()

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
    edge_labels = {}
    
    for src in active_nodes:
        for dst in active_nodes:
            prob = matrix.loc[src, dst]
            if prob >= min_prob_threshold:
                active_edges.append((src, dst, prob))
                edge_labels[(src, dst)] = f"{prob:.2f}"
                
    draw_edges(active_edges)
    
    # 4. Annotate Probabilities (only for the solid foreground edges)
    if edge_labels:
        nx.draw_networkx_edge_labels(
            G_dummy, pos, ax=ax, edge_labels=edge_labels,
            font_size=8, font_weight='bold', font_color='red', label_pos=0.3,
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.75)
        )

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axis('off')


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
        birdsong_title = f"Birdsong\n(Manhattan Dist: {birdsong_manhattan:.2f})"
    else:
        tweety_title = "TweetyBERT"
        birdsong_title = "Birdsong"

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