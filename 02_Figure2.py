"""
Figure 2 – Area-weighted tree-cover transition matrices (2000 → 2023)

Each panel shows how forest pixels moved between 10%-wide bins of tree-cover
percentage (TC%) from 2000 to 2023.  Rows are the starting bin in 2000, columns
are the ending bin in 2023.  The background colour encodes the total forest area
(km², log scale) that made each transition; the numbers annotated on each cell
show what percentage of that 2000-bin's area ended up in that 2023 bin, so every
row sums to 100%.  Pixels sitting on the diagonal didn't change bin; pixels below
it lost tree cover.

Area weighting is important here because pixels near the equator cover less ground
than pixels at the same lat/lon spacing near the poles — we account for this using
the standard spherical-cap formula.

Data required
-------------
target_ifl.npy   : float32 array of shape (n_years, n_pixels) with annual
                   Hansen-derived tree-cover values at 0.05° resolution,
                   covering 2000–2023 (24 years).
biomes_kg_005.npy: int array (one value per pixel) with Köppen-Geiger biome class
                   (1 = Tropics, 2 = Arid, 3 = Temperate, 4 = Boreal).

Set DATA_DIR and OUTPUT_PATH below before running.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm

# ---------------------------------------------------------------------------
# Paths – adjust these before running
# ---------------------------------------------------------------------------
DATA_DIR    = r"E:\python\TreeCoverDataReview"
OUTPUT_PATH = r"Figure2.pdf"
SAVE_FIGURE = True

# ---------------------------------------------------------------------------
# Load and prepare data
# ---------------------------------------------------------------------------
data   = np.load(f"{DATA_DIR}/target_ifl.npy")        # (n_years, H, W)
biomes = np.load(f"{DATA_DIR}/biomes_kg_005.npy").reshape(-1)

print(f"Tree-cover array:  {data.shape}")
print(f"Biome pixel count: {biomes.shape}")

# Mask pixels below 10% — we're only interested in areas that qualify as forest.
TC_THRESHOLD = 10
data = np.where(data < TC_THRESHOLD, np.nan, data)

# Collapse the spatial grid into a single pixel dimension for easier indexing.
data_flat = data.reshape(data.shape[0], -1)   # (n_years, n_pixels)

# TC% bins: [10,20), [20,30), …, [90,100]
bins       = np.arange(10, 101, 10)
bin_labels = [f"{bins[i]}-{bins[i+1]-1}" for i in range(len(bins) - 1)]

# The two snapshots we compare: start and end of the record.
data_2000 = data_flat[0]
data_2023 = data_flat[-1]

biome_dict = {
    "Tropics":   1,
    "Arid":      2,
    "Temperate": 3,
    "Boreal":    4,
}

# ---------------------------------------------------------------------------
# Latitude index array for area weighting
#
# The global grid is 3600 lat-bands × 7200 lon-bands at 0.05° resolution.
# We need to know which latitude band each flattened pixel belongs to so we
# can look up its area weight later.
# ---------------------------------------------------------------------------
N_LAT = 3600
N_LON = 7200
lat_indices = np.repeat(np.arange(N_LAT), N_LON)   # shape: (n_pixels,)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def compute_pixel_area_lat(resolution=0.05):
    """
    Area of one grid cell as a function of latitude (in km²).

    For an equal-angle grid the cell width in the east-west direction shrinks
    with the cosine of latitude, so tropical pixels cover less ground than the
    same-sized box near the equator.  The formula used is the standard
    spherical-cap integral: A = R² × |sin(φ₂) − sin(φ₁)| × Δλ.
    """
    R = 6371.0   # Earth radius in km
    lat_edges     = np.arange(-90, 90 + resolution, resolution)
    lat_edges_rad = np.deg2rad(lat_edges)
    lon_res_rad   = np.deg2rad(resolution)

    area_per_lat = (R ** 2) * np.abs(
        np.sin(lat_edges_rad[1:]) - np.sin(lat_edges_rad[:-1])
    ) * lon_res_rad

    return area_per_lat


def compute_weighted_transition_matrix(data_t0, data_t1, bins, lat_idx,
                                       resolution=0.05, valid_mask=None):
    """
    Build the area-weighted transition matrix between two time points.

    Each cell (i, j) accumulates the total km² of pixels that were in TC bin i
    in t0 and ended up in TC bin j in t1.  We also return the raw pixel counts
    in case you need them for sanity checks.
    """
    n_bins = len(bins) - 1
    area_per_lat = compute_pixel_area_lat(resolution)

    # Assign each pixel to a bin, marking out-of-range ones as -1.
    bins_t0 = np.digitize(data_t0, bins, right=False) - 1
    bins_t1 = np.digitize(data_t1, bins, right=False) - 1
    bins_t0 = np.where((bins_t0 >= 0) & (bins_t0 < n_bins), bins_t0, -1)
    bins_t1 = np.where((bins_t1 >= 0) & (bins_t1 < n_bins), bins_t1, -1)

    # Only pixels that have valid TC values in both years contribute.
    base_mask = (bins_t0 >= 0) & (bins_t1 >= 0) & \
                (~np.isnan(data_t0)) & (~np.isnan(data_t1))
    if valid_mask is not None:
        base_mask = base_mask & valid_mask

    pixel_areas = area_per_lat[lat_idx]

    tm_area  = np.zeros((n_bins, n_bins), dtype=float)
    tm_count = np.zeros((n_bins, n_bins), dtype=int)

    for i in range(n_bins):
        for j in range(n_bins):
            mask_ij = base_mask & (bins_t0 == i) & (bins_t1 == j)
            tm_count[i, j] = np.sum(mask_ij)
            tm_area[i, j]  = np.sum(pixel_areas[mask_ij])

    return tm_area, tm_count


def analyze_transition_distances_weighted(tm_area):
    """
    Summarise the transition matrix into four broad categories based on how
    far pixels moved (measured in number of 10%-bins):
      - No change  : same bin in both years (diagonal)
      - Low (±1)   : moved to an adjacent bin
      - Mid (±2–3) : moved two or three bins
      - High (>±3) : large shift, more than three bins
    All percentages are relative to the total forest area in the matrix.
    """
    n_bins     = tm_area.shape[0]
    total_area = np.sum(tm_area)

    if total_area == 0:
        return {k: 0 for k in
                ('no_change', 'low_change', 'mid_change', 'high_change', 'total_area_km2')}

    no_change = np.sum(np.diag(tm_area))
    adjacent  = sum(tm_area[i, j] for i in range(n_bins) for j in range(n_bins)
                    if abs(i - j) == 1)
    gradual   = sum(tm_area[i, j] for i in range(n_bins) for j in range(n_bins)
                    if 2 <= abs(i - j) <= 3)
    abrupt    = sum(tm_area[i, j] for i in range(n_bins) for j in range(n_bins)
                    if abs(i - j) > 3)

    return {
        'no_change':             100 * no_change / total_area,
        'low_change':            100 * adjacent  / total_area,
        'mid_change':            100 * gradual   / total_area,
        'high_change':           100 * abrupt    / total_area,
        'total_area_km2':        total_area,
        'no_change_area_km2':    no_change,
        'low_change_area_km2':   adjacent,
        'mid_change_area_km2':   gradual,
        'high_change_area_km2':  abrupt,
    }


# ---------------------------------------------------------------------------
# Compute transition matrices
# ---------------------------------------------------------------------------
print("Computing area-weighted transition matrices (this may take a minute)…")

transition_global_area, _ = compute_weighted_transition_matrix(
    data_2000, data_2023, bins, lat_indices
)
stats_global_area = analyze_transition_distances_weighted(transition_global_area)

print(f"\nGlobal — total forest area: {stats_global_area['total_area_km2']/1e6:.2f} M km²")
print(f"  No change : {stats_global_area['no_change']:.1f}%")
print(f"  Low (±1)  : {stats_global_area['low_change']:.1f}%")
print(f"  Mid (2–3) : {stats_global_area['mid_change']:.1f}%")
print(f"  High (>3) : {stats_global_area['high_change']:.1f}%")

transitions_by_biome_area = {}
stats_by_biome_area       = {}

for biome_name, biome_value in biome_dict.items():
    biome_mask = (biomes == biome_value)
    tm_area, _ = compute_weighted_transition_matrix(
        data_2000, data_2023, bins, lat_indices, valid_mask=biome_mask
    )
    transitions_by_biome_area[biome_name] = tm_area
    stats_by_biome_area[biome_name]       = analyze_transition_distances_weighted(tm_area)

    s = stats_by_biome_area[biome_name]
    print(f"\n{biome_name} — total forest area: {s['total_area_km2']/1e6:.2f} M km²")
    print(f"  No change : {s['no_change']:.1f}%")
    print(f"  Low (±1)  : {s['low_change']:.1f}%")
    print(f"  Mid (2–3) : {s['mid_change']:.1f}%")
    print(f"  High (>3) : {s['high_change']:.1f}%")

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size']   = 10
plt.rcParams['pdf.fonttype'] = 42

CBAR_SHRINK       = 0.75
CBAR_ASPECT       = 25
CBAR_PAD          = 0.03
CBAR_TICK_SIZE    = 9
MATRIX_TICK_SIZE  = 10
MATRIX_LABEL_SIZE = 12
TITLE_SIZE        = 10
BAR_WIDTH         = 0.15
BAR_TICK_SIZE     = 12


def normalize_matrix_by_row(matrix):
    """
    Express each cell as a percentage of its row's total area, so reading
    across a row tells you where the pixels that started in that 2000-bin
    ended up by 2023.  Rows sum to 100 %.
    """
    row_sums = matrix.sum(axis=1, keepdims=True)
    return np.where(row_sums > 0, (matrix / row_sums) * 100, np.nan)


def draw_matrix(ax, matrix, title, show_xlabel=True, show_ylabel=True):
    """
    Draw one transition-matrix panel.

    Background colour: raw area in km² on a log scale (cividis colormap).
    Annotations: row-normalised percentages.  Cells below 1 % are left blank
    to avoid cluttering the plot with noise.  A dashed diagonal marks the
    "no change" line.
    """
    im = ax.imshow(matrix, cmap='cividis',
                   norm=LogNorm(vmin=1, vmax=matrix.max()))

    norm_pct = normalize_matrix_by_row(matrix)

    for i in range(norm_pct.shape[0]):
        for j in range(norm_pct.shape[1]):
            val = norm_pct[i, j]
            if np.isnan(val) or val < 1:
                continue
            # Diagonal cells tend to be bright yellow in cividis, so dark text
            # reads better there; use white everywhere else.
            color = 'black' if i == j else 'white'
            ax.text(j, i, f'{int(round(val))}',
                    ha='center', va='center',
                    fontsize=10, color=color, fontweight='bold')

    ax.set_xticks(range(len(bin_labels)))
    ax.set_yticks(range(len(bin_labels)))
    ax.set_xticklabels(bin_labels, rotation=45, ha='right',
                       fontsize=MATRIX_TICK_SIZE)
    ax.set_yticklabels(bin_labels, fontsize=MATRIX_TICK_SIZE)

    if show_xlabel:
        ax.set_xlabel('TC 2023 (%)', fontsize=MATRIX_LABEL_SIZE, fontweight='bold')
    if show_ylabel:
        ax.set_ylabel('TC 2000 (%)', fontsize=MATRIX_LABEL_SIZE, fontweight='bold')

    ax.set_title(title, fontweight='bold', loc='left', fontsize=TITLE_SIZE)

    # Dashed diagonal as a visual reference for "no net change"
    ax.plot([-0.5, len(bin_labels) - 0.5],
            [-0.5, len(bin_labels) - 0.5],
            'k--', linewidth=1, alpha=0.5)

    cbar = plt.colorbar(im, ax=ax, shrink=CBAR_SHRINK,
                        aspect=CBAR_ASPECT, pad=CBAR_PAD)
    cbar.set_label('Area (km², log scale)', fontsize=MATRIX_LABEL_SIZE - 1)
    cbar.ax.tick_params(labelsize=CBAR_TICK_SIZE)


fig = plt.figure(figsize=(18, 11), dpi=500)
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.25, wspace=0.15)

# Transition matrix panels (a–e)
draw_matrix(fig.add_subplot(gs[0, 0]), transition_global_area,                'a  Global')
draw_matrix(fig.add_subplot(gs[0, 1]), transitions_by_biome_area['Tropics'],  'b  Tropics',   show_ylabel=False)
draw_matrix(fig.add_subplot(gs[0, 2]), transitions_by_biome_area['Arid'],     'c  Arid',      show_ylabel=False)
draw_matrix(fig.add_subplot(gs[1, 0]), transitions_by_biome_area['Temperate'],'d  Temperate')
draw_matrix(fig.add_subplot(gs[1, 1]), transitions_by_biome_area['Boreal'],   'e  Boreal',    show_ylabel=False)

# Bar chart summarising the four change categories across all regions (panel f)
ax_bar = fig.add_subplot(gs[1, 2])

categories = ['No change', 'Low\n(±10%)', 'Mid\n(±20–30%)', 'High\n(>±30%)']
x          = np.arange(len(categories))
regions    = ['Global'] + list(biome_dict.keys())
all_stats  = [stats_global_area] + [stats_by_biome_area[b] for b in biome_dict]
greys      = ['#2b2b2b', '#555555', '#7f7f7f', '#aaaaaa', '#d4d4d4']

for i, (region, stats, grey) in enumerate(zip(regions, all_stats, greys)):
    values = [stats['no_change'], stats['low_change'],
              stats['mid_change'], stats['high_change']]
    offset = BAR_WIDTH * (i - len(regions) / 2 + 0.5)
    ax_bar.bar(x + offset, values, BAR_WIDTH,
               label=region, color=grey, alpha=0.9, edgecolor='white')

ax_bar.set_xlabel('Degree of change', fontsize=MATRIX_LABEL_SIZE, fontweight='bold')
ax_bar.set_ylabel('Percentage of Forest Area (%)', fontsize=MATRIX_LABEL_SIZE)
ax_bar.set_xticks(x)
ax_bar.set_xticklabels(categories, fontsize=BAR_TICK_SIZE)
ax_bar.tick_params(axis='y', labelsize=BAR_TICK_SIZE)
ax_bar.set_ylim(bottom=0)
ax_bar.grid(axis='y', alpha=0.3)
ax_bar.set_title('f', fontweight='bold', loc='left', fontsize=TITLE_SIZE)
ax_bar.legend(frameon=True, loc='upper right', fontsize=10)

# ---------------------------------------------------------------------------
# Save / show
# ---------------------------------------------------------------------------
if SAVE_FIGURE:
    fig.savefig(OUTPUT_PATH, bbox_inches='tight')
    print(f"\nFigure saved to {OUTPUT_PATH}")

plt.show()
