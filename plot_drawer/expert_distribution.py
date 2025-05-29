import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

argparser = argparse.ArgumentParser(description="Plot expert distribution heatmap from CSV file.")
argparser.add_argument("--csv_file", type=str, required=True, help="Path to the CSV file containing expert distribution data.")
argparser.add_argument("--output_dir", type=str, default="", help="Path to save the output heatmap image. If empty, will not save.")
args = argparser.parse_args()

# Read CSV path from command-line argument
output_file_name = args.csv_file.split("/")[-1].replace(".csv", ".png")
final_output_path = args.output_dir + output_file_name

# Read CSV
df = pd.read_csv(args.csv_file)

# Get dimensions
num_layers = df.shape[0]
num_experts = df.shape[1] - 1

# Transpose if layers >= experts (i.e., experts on y-axis, layers on x-axis)
if num_experts <= num_layers:
    heatmap_data = df.set_index("layer_id").T.sort_index(ascending=False, key=lambda idx: idx.map(lambda x: int(x.split('_')[1])))
    # Rename expert axis for display
    heatmap_data.index = heatmap_data.index.map(lambda x: int(x.split('_')[1]))
    x_label = "Layer ID"
    y_label = "Expert ID"
else:
    heatmap_data = df.set_index("layer_id").sort_index(ascending=False)
    heatmap_data.columns = heatmap_data.columns.map(lambda x: int(x.split('_')[1]))
    x_label = "Expert ID"
    y_label = "Layer ID"

layout_ratio = num_experts / num_layers if num_experts > num_layers else num_layers / num_experts
layout_x = 12
layout_y = int(layout_x / layout_ratio)
# Plot heatmap
fig = plt.figure(figsize=(layout_x, layout_y))
ax = sns.heatmap(
    heatmap_data,
    cmap="Greens",
    linecolor='gray',
    square=True,
    cbar_kws={"shrink": 0.5},
)
colorbar = ax.collections[0].colorbar
colorbar.ax.tick_params(labelsize=10)
colorbar.formatter.set_useOffset(False)
colorbar.formatter.set_scientific(False)
colorbar.update_ticks()

plt.title("Token Count Heatmap", fontsize=14)
plt.xlabel(x_label, fontsize=14)
plt.ylabel(y_label, fontsize=14)
plt.tight_layout()
plt.show()
fig.savefig(final_output_path, bbox_inches='tight', dpi=300)
