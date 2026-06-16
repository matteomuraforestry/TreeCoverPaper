"""
05_tc_dynamics.py

Compute area-weighted tree cover transition matrices between 2000 and 2023.

Pixels are classified into 10-percentage-point TC bins (10-20 %, 20-30 %, …,
90-100 %).  Transitions between bins are categorised as:
    no change   — same bin
    adjacent    — ±1 bin
    gradual     — 2-3 bins
    abrupt      — >3 bins

Pixel areas are latitude-corrected to account for the decreasing area of
equal-angle grid cells toward the poles.

Inputs:
    target_ifl.npy      — Hansen TC [years x lat x lon], TC ≥ 10 %
    biomes_kg_005.npy   — Koppen-Geiger biome map at 0.05°
Outputs:
    Transition matrix plots and bar charts (displayed on screen).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

DATA_DIR = r""
TC_THRESHOLD = 10
N_LAT, N_LON = 3600, 7200  # grid dimensions at 0.05°

# ----- load -----
data = np.load(os.path.join(DATA_DIR, "target_ifl.npy"))
data = np.where(data < TC_THRESHOLD, np.nan, data)
print(f"Data shape: {data.shape}")

biomes = np.load(os.path.join(DATA_DIR, "biomes_kg_005.npy")).reshape(-1)

data_flat = data.reshape(data.shape[0], -1)
data_2000 = data_flat[0]
data_2023 = data_flat[-1]

# latitude index for each flattened pixel (used for area weighting)
lat_indices = np.repeat(np.arange(N_LAT), N_LON)

bins = np.arange(10, 101, 10)
bin_labels = [f"{bins[i]}-{bins[i+1]-1}" for i in range(len(bins) - 1)]


# ----- area weighting -----

def compute_pixel_area_lat(resolution=0.05):
    """Pixel area (km²) as a function of latitude for an equal-angle grid."""
    R = 6371.0
    lat_edges = np.arange(-90, 90 + resolution, resolution)
    lat_edges_rad = np.deg2rad(lat_edges)
    lon_res_rad = np.deg2rad(resolution)
    area = R ** 2 * np.abs(
        np.sin(lat_edges_rad[1:]) - np.sin(lat_edges_rad[:-1])
    ) * lon_res_rad
    return area


AREA_PER_LAT = compute_pixel_area_lat()


def compute_weighted_transition_matrix(data_t0, data_t1, bins, lat_idx,
                                       valid_mask=None):
    """
    Area-weighted transition matrix between two TC arrays.

    Returns
    -------
    matrix_area  : (n_bins, n_bins) — transitions summed in km²
    matrix_count : (n_bins, n_bins) — transitions as pixel count
    """
    n_bins = len(bins) - 1
    matrix_area  = np.zeros((n_bins, n_bins))
    matrix_count = np.zeros((n_bins, n_bins), dtype=int)

    bins_t0 = np.digitize(data_t0, bins, right=False) - 1
    bins_t1 = np.digitize(data_t1, bins, right=False) - 1
    bins_t0 = np.where((bins_t0 >= 0) & (bins_t0 < n_bins), bins_t0, -1)
    bins_t1 = np.where((bins_t1 >= 0) & (bins_t1 < n_bins), bins_t1, -1)

    base_valid = (
        (bins_t0 >= 0) & (bins_t1 >= 0)
        & ~np.isnan(data_t0) & ~np.isnan(data_t1)
    )
    if valid_mask is not None:
        base_valid = base_valid & valid_mask

    pixel_areas = AREA_PER_LAT[lat_idx]

    for i in range(n_bins):
        for j in range(n_bins):
            mask_ij = base_valid & (bins_t0 == i) & (bins_t1 == j)
            matrix_count[i, j] = np.sum(mask_ij)
            matrix_area[i, j]  = np.sum(pixel_areas[mask_ij])

    return matrix_area, matrix_count


def transition_stats(matrix_area):
    """Fraction of total forest area for each transition category."""
    n = matrix_area.shape[0]
    total = np.sum(matrix_area)
    if total == 0:
        return {k: 0 for k in ("no_change", "adjacent", "gradual", "abrupt",
                                "total_area_km2")}
    no_change = np.sum(np.diag(matrix_area))
    adjacent  = sum(matrix_area[i, j]
                    for i in range(n) for j in range(n) if abs(i - j) == 1)
    gradual   = sum(matrix_area[i, j]
                    for i in range(n) for j in range(n) if 2 <= abs(i - j) <= 3)
    abrupt    = sum(matrix_area[i, j]
                    for i in range(n) for j in range(n) if abs(i - j) > 3)
    return {
        "no_change": 100 * no_change / total,
        "adjacent":  100 * adjacent  / total,
        "gradual":   100 * gradual   / total,
        "abrupt":    100 * abrupt    / total,
        "total_area_km2":      total,
        "no_change_area_km2":  no_change,
        "adjacent_area_km2":   adjacent,
        "gradual_area_km2":    gradual,
        "abrupt_area_km2":     abrupt,
    }


# ----- compute transitions (global + biomes) -----
print("Computing area-weighted transition matrices (2000 → 2023) …")

trans_global_area, _ = compute_weighted_transition_matrix(
    data_2000, data_2023, bins, lat_indices
)
stats_global = transition_stats(trans_global_area)

print(f"\nGlobal — total forest area: {stats_global['total_area_km2'] / 1e6:.2f} M km²")
for key in ("no_change", "adjacent", "gradual", "abrupt"):
    print(f"  {key:12s}: {stats_global[key]:.1f}%")

biome_dict = {"Tropics": 1, "Arid": 2, "Temperate": 3, "Boreal": 4}
trans_by_biome = {}
stats_by_biome = {}

for biome_name, biome_code in biome_dict.items():
    mask = biomes == biome_code
    trans_area, _ = compute_weighted_transition_matrix(
        data_2000, data_2023, bins, lat_indices, valid_mask=mask
    )
    trans_by_biome[biome_name] = trans_area
    stats_by_biome[biome_name] = transition_stats(trans_area)
    s = stats_by_biome[biome_name]
    print(f"\n{biome_name} — {s['total_area_km2'] / 1e6:.2f} M km²  |  "
          f"no-change {s['no_change']:.1f}%  adj {s['adjacent']:.1f}%  "
          f"grad {s['gradual']:.1f}%  abrupt {s['abrupt']:.1f}%")


# ----- visualise: heatmaps -----
fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=150)

def _plot_heatmap(ax, matrix, title):
    im = ax.imshow(matrix, cmap="cividis",
                   norm=LogNorm(vmin=1, vmax=matrix.max()))
    ax.set_xticks(range(len(bin_labels)))
    ax.set_yticks(range(len(bin_labels)))
    ax.set_xticklabels(bin_labels, rotation=45, ha="right")
    ax.set_yticklabels(bin_labels)
    ax.set_xlabel("TC 2023 (%)")
    ax.set_ylabel("TC 2000 (%)")
    ax.set_title(title, fontweight="bold", loc="left")
    ax.plot([-0.5, len(bin_labels) - 0.5], [-0.5, len(bin_labels) - 0.5],
            "k--", linewidth=1.5, alpha=0.5)
    plt.colorbar(im, ax=ax, label="Area (km², log scale)")

_plot_heatmap(axes[0, 0], trans_global_area, "(a) Global")
for idx, (name, code) in enumerate(biome_dict.items()):
    row, col = (0, idx + 1) if idx < 2 else (1, idx - 2)
    _plot_heatmap(axes[row, col], trans_by_biome[name],
                  f"({chr(98 + idx)}) {name}")

axes[1, 2].axis("off")
plt.tight_layout()
plt.show()

# ----- visualise: bar chart -----
fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

categories = ["No Change", "Adjacent\n(±1 bin)", "Gradual\n(2-3 bins)", "Abrupt\n(>3 bins)"]
x      = np.arange(len(categories))
width  = 0.15
regions   = ["Global"] + list(biome_dict.keys())
all_stats = [stats_global] + [stats_by_biome[b] for b in biome_dict]
greys     = ["#2b2b2b", "#555555", "#7f7f7f", "#aaaaaa", "#d4d4d4"]

for i, (region, s, grey) in enumerate(zip(regions, all_stats, greys)):
    values = [s["no_change"], s["adjacent"], s["gradual"], s["abrupt"]]
    offset = width * (i - len(regions) / 2 + 0.5)
    ax.bar(x + offset, values, width, label=region, color=grey, alpha=0.9,
           edgecolor="white")

ax.set_xlabel("Transition Type", fontsize=12)
ax.set_ylabel("Percentage of Forest Area (%)", fontsize=12)
ax.set_title("Area-Weighted TC Transition Patterns (2000→2023)",
             fontsize=13, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend(title="Region")
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(bottom=0)
plt.tight_layout()
plt.show()
