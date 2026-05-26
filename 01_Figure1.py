"""
Figure 1 – Global and biome-level shifts in tree-cover distribution (2000–2023)

Each panel is a heatmap whose rows are years and columns are 10 %-wide bins of
tree-cover percentage (TC%).  Colours encode how many pixels have moved into or
out of each bin relative to the year-2000 baseline, so a green cell means "more
pixels in that bin than in 2000" and a brown cell means "fewer pixels".

Data required
-------------
target_ifl.npy   : float32 array of shape (n_years, n_pixels) containing annual
                   Hansen-derived tree-cover values at 0.05° resolution,
                   covering 2000–2023 (24 years).
biomes_kg_005.npy: int array with one value per pixel encoding the Köppen-Geiger
                   biome class (1 = Tropics, 2 = Arid, 3 = Temperate, 4 = Boreal).

Set DATA_DIR below to the folder that holds both .npy files, and OUTPUT_PATH to
wherever you want the PDF saved.  Set SAVE_FIGURE = False if you just want the
plot on screen without saving.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Paths – adjust these two lines before running
# ---------------------------------------------------------------------------
DATA_DIR    = r"E:\python\TreeCoverDataReview"   # folder containing the .npy files
OUTPUT_PATH = r"Figure1.pdf"                     # where to save the finished figure
SAVE_FIGURE = True                               # set False to only display on screen

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
data   = np.load(f"{DATA_DIR}/target_ifl.npy")        # shape: (n_years, H, W)
biomes = np.load(f"{DATA_DIR}/biomes_kg_005.npy").reshape(-1)

print(f"Tree-cover array:  {data.shape}")
print(f"Biome pixel count: {biomes.shape}")

# ---------------------------------------------------------------------------
# Mask pixels below the minimum tree-cover threshold.
# We treat anything under 10 % as "not a forest pixel" so the histograms
# capture only the meaningful part of the distribution.
# ---------------------------------------------------------------------------
TC_THRESHOLD = 10
data = np.where(data < TC_THRESHOLD, np.nan, data)

# Flatten the spatial dimensions so we can histogram across all pixels at once.
data_flat = data.reshape(data.shape[0], -1)   # shape: (n_years, n_pixels)

# ---------------------------------------------------------------------------
# Histograms and year-2000 baseline difference
# ---------------------------------------------------------------------------
# One bin per 10 % increment → 9 bins covering [10, 20), [20, 30), …, [90, 100]
bins = np.arange(10, 101, 10)

# Count pixels per TC% bin for every year.
counts = np.stack(
    [np.histogram(data_flat[i], bins=bins)[0] for i in range(data_flat.shape[0])]
)   # shape: (n_years, n_bins)

# Subtract the year-2000 snapshot from every year.
# A positive value means that bin gained pixels; negative means it lost them.
year2000_diff = counts - counts[0]

years = np.arange(2000, 2024)

# ---------------------------------------------------------------------------
# Biome definitions (Köppen-Geiger broad classes)
# ---------------------------------------------------------------------------
biome_dict = {
    "Tropics":    1,
    "Arid":       2,
    "Temperate":  3,
    "Boreal":     4,
}

# Pre-compute the biome-level diffs so we only loop over pixels once.
all_biome_diffs = []
for biome_name, biome_value in biome_dict.items():
    biome_mask   = (biomes == biome_value)
    biome_counts = np.stack(
        [np.histogram(data_flat[i][biome_mask], bins=bins)[0]
         for i in range(data_flat.shape[0])]
    )
    all_biome_diffs.append(biome_counts - biome_counts[0])

# ---------------------------------------------------------------------------
# Figure layout
# ---------------------------------------------------------------------------
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size']   = 10
plt.rcParams['pdf.fonttype'] = 42    # ensures fonts are embedded properly in PDFs

cm = 1 / 2.54   # conversion factor for setting figure size in centimetres
RESOLUTION = 500

MATRIX_TICK_SIZE  = 10
MATRIX_LABEL_SIZE = 12
TITLE_SIZE        = 10
CBAR_TICK_SIZE    = 9

# Only label a handful of years on the y-axis; the rest stay blank to avoid clutter.
target_years  = [2000, 2005, 2010, 2015, 2020]
ytick_labels  = [str(y) if y in target_years else '' for y in years]
biome_letters = ['b', 'c', 'd', 'e']

fig = plt.figure(figsize=(18, 11), dpi=RESOLUTION)

# The global panel is roughly 20 % wider than each biome panel because it spans
# both rows of the 2 × 2 biome grid, so we give it a slightly higher width ratio.
gs = GridSpec(2, 3, figure=fig,
              width_ratios=[1.2, 1, 1],
              hspace=0.25, wspace=0.35)

# ---------------------------------------------------------------------------
# Panel a – Global
# ---------------------------------------------------------------------------
ax_global = fig.add_subplot(gs[:, 0])   # span both rows

sns.heatmap(
    year2000_diff,
    xticklabels=bins[:-1],
    yticklabels=ytick_labels,
    cmap="BrBG",
    center=0,
    ax=ax_global,
    cbar_kws={
        "label": "Difference in Number of Pixels\nsince 2000",
        "shrink": 0.9,
        "orientation": "horizontal",
        "pad": 0.18,
    },
)

# A thin border around the heatmap makes it easier to read when printed.
ax_global.add_patch(
    Rectangle((0, 0), year2000_diff.shape[1], year2000_diff.shape[0],
              fill=False, edgecolor='black', linewidth=1, clip_on=False)
)

ax_global.set_xlabel("Bins of TC%", fontsize=MATRIX_LABEL_SIZE, fontweight='bold')
ax_global.set_ylabel("Year",        fontsize=MATRIX_LABEL_SIZE, fontweight='bold')
ax_global.tick_params(axis='y', labelsize=MATRIX_TICK_SIZE)
plt.setp(ax_global.get_xticklabels(), rotation=0, ha='center', fontsize=MATRIX_TICK_SIZE)
ax_global.set_title('a  Global', fontweight='bold', loc='left', fontsize=TITLE_SIZE)

# Use scientific notation on the global colorbar and nudge the ×10^n label
# so it doesn't overlap with the axis ticks.
cbar_global = ax_global.collections[0].colorbar
cbar_global.ax.tick_params(rotation=0, labelsize=CBAR_TICK_SIZE)
cbar_global.formatter = ticker.ScalarFormatter(useMathText=True)
cbar_global.formatter.set_powerlimits((0, 0))
cbar_global.update_ticks()
offset = cbar_global.ax.xaxis.get_offset_text()
offset.set_position((1, -2))
offset.set_verticalalignment('bottom')
offset.set_horizontalalignment('center')

# ---------------------------------------------------------------------------
# Panels b–e – individual biomes arranged in a 2 × 2 grid
# ---------------------------------------------------------------------------
for idx, (biome_name, biome_value) in enumerate(biome_dict.items()):
    row = idx // 2
    col = (idx % 2) + 1
    ax  = fig.add_subplot(gs[row, col])

    biome_year2000_diff = all_biome_diffs[idx]

    sns.heatmap(
        biome_year2000_diff,
        xticklabels=bins[:-1],
        yticklabels=ytick_labels,
        cmap="BrBG",
        center=0,
        ax=ax,
        cbar_kws={"shrink": 0.7, "pad": 0.05},
    )

    ax.add_patch(
        Rectangle((0, 0), biome_year2000_diff.shape[1], biome_year2000_diff.shape[0],
                  fill=False, edgecolor='black', linewidth=1, clip_on=False)
    )

    # Scientific notation on biome colorbars (vertical orientation)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(rotation=0, labelsize=CBAR_TICK_SIZE)
    cbar.formatter = ticker.ScalarFormatter(useMathText=True)
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()
    offset = cbar.ax.yaxis.get_offset_text()
    offset.set_position((2, 1.0))
    offset.set_verticalalignment('center')
    offset.set_horizontalalignment('left')

    # Only the bottom row needs an x-axis label; duplicating it on the top row
    # would waste space and look cluttered.
    if row == 0:
        ax.set_xlabel("")
    else:
        ax.set_xlabel("Bins of TC%", fontsize=MATRIX_LABEL_SIZE, fontweight='bold')

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right',
             rotation_mode='anchor', fontsize=MATRIX_TICK_SIZE)

    ax.set_ylabel("", fontsize=MATRIX_LABEL_SIZE)
    ax.tick_params(axis='y', labelsize=CBAR_TICK_SIZE, rotation=0)
    ax.set_title(f'{biome_letters[idx]}  {biome_name}',
                 fontweight='bold', loc='left', fontsize=TITLE_SIZE)

# ---------------------------------------------------------------------------
# Save / show
# ---------------------------------------------------------------------------
if SAVE_FIGURE:
    fig.savefig(OUTPUT_PATH, bbox_inches='tight')
    print(f"Figure saved to {OUTPUT_PATH}")

plt.show()
