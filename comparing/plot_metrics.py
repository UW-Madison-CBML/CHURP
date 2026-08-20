import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# 1. Load the Data from the CSV file
df = pd.read_csv(sys.argv[1])

# Assign bird names to the index (Bird 0, Bird 1, etc.)
df.index = [f"Bird {i}" for i in range(len(df))]

# 2. Configure Plot Styling
sns.set_theme(style="whitegrid", rc={"axes.edgecolor": ".8"})
plt.rcParams['font.family'] = 'sans-serif'

# Define a professional color palette (Blues for Tweety, Oranges/Reds for Birdsong)
colors = {
    'Tweety': '#1f77b4',    # Standard Blue
    'Birdsong': '#d62728',  # Standard Red
    'Audio': '#2ca02c'        # Green
}

# 3. Create Figure and Subplots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 12))
fig.suptitle("Model Performance Comparison: Tweety vs. Birdsong", fontsize=18, fontweight='bold', y=0.98)

# Helper function to plot grouped bar charts cleanly
def plot_grouped_bar(ax, cols, title, y_label):
    # Reorder columns to group Set 1 and Set 2 logically
    plot_df = df[cols]
    
    # Map colors to the specific columns passed
    col_colors = [colors['Tweety'], colors['Birdsong']]
    
    plot_df.plot(kind='bar', ax=ax, color=col_colors, width=0.8, edgecolor='black', linewidth=0.5)
    ax.set_title(title, fontsize=14, pad=10)
    ax.set_ylabel(y_label, fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    
    # Clean up legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, ['Tweety', 'Birdsong'], 
              loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)

# Top-Left: Frame Error Rate (FER)
plot_grouped_bar(
    axes[0, 0], 
    ['Tweety FER', 'Birdsong FER'], 
    'Frame Error Rate (FER) Lower is Better', 
    'Error Rate'
)

# Top-Right: Entropy Correlation
plot_grouped_bar(
    axes[0, 1], 
    ['Tweety Entropy Cor', 'Birdsong Entropy Cor'], 
    'Entropy Correlation (Higher is Better)', 
    'Correlation Coefficient'
)

# Bottom-Left: Duration Correlation
plot_grouped_bar(
    axes[1, 0], 
    ['Tweety Duration Cor', 'Birdsong Duration Cor'], 
    'Duration Correlation (Higher is Better)', 
    'Correlation Coefficient'
)

# Bottom-Right: Audio Minutes
df['Audio Minutes'].plot(
    kind='bar', 
    ax=axes[1, 1], 
    color=colors['Audio'], 
    edgecolor='black', 
    linewidth=0.5
)
axes[1, 1].set_title('Audio Minutes per Bird', fontsize=14, pad=10)
axes[1, 1].set_ylabel('Minutes', fontsize=12)
axes[1, 1].tick_params(axis='x', rotation=45)

# 4. Final Layout Adjustments
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Leave room for the main title
plt.subplots_adjust(hspace=0.4) # Add vertical space for legends/x-axis labels

# Save and show
plt.savefig("bird_model_comparison.png", dpi=600, bbox_inches='tight')
plt.show()
