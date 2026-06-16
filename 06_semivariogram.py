"""
06_semivariogram.py

Estimate spatial autocorrelation length via empirical semivariogram analysis.
Uses the variogram range (distance at which semivariance reaches 95% of the
sill) as a proxy for the autocorrelation length scale.

Results are used later to compute effective sample sizes for confidence
intervals and disequilibrium estimates.

Inputs:
    target_ifl.npy      — Hansen TC [years x lat x lon], TC ≥ 10 %
    biomes_kg_005.npy   — Koppen-Geiger biome map at 0.05°
Outputs:
    Printed autocorrelation lengths (pixels and km) per biome.
    Semivariogram plots displayed on screen.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist

DATA_DIR = r""
RESOLUTION_KM = 5.0   # approximate pixel size at equator for 0.05° grid


def estimate_autocorr_length(data_2d, max_dist_pixels=200, n_samples=3000):
    """
    Estimate autocorrelation length from an empirical semivariogram.

    Parameters
    ----------
    data_2d        : 2-D array (lat × lon) — single time slice
    max_dist_pixels: upper distance limit for variogram bins
    n_samples      : random pixels to sample (speed vs accuracy)

    Returns
    -------
    autocorr_length : int — distance (pixels) where semivariance reaches
                      95% of the sill
    """
    valid_rows, valid_cols = np.where(~np.isnan(data_2d))

    if len(valid_rows) > n_samples:
        idx = np.random.choice(len(valid_rows), n_samples, replace=False)
        valid_rows = valid_rows[idx]
        valid_cols = valid_cols[idx]

    coords = np.column_stack([valid_rows, valid_cols])
    values = data_2d[valid_rows, valid_cols]

    # pairwise distances and squared differences (= 2 × semivariance)
    distances   = pdist(coords, metric="euclidean")
    sq_diffs    = pdist(values.reshape(-1, 1), metric="sqeuclidean")

    bins = np.arange(0, max_dist_pixels, 5)
    digitized = np.digitize(distances, bins)

    semivariances = []
    bin_centers   = []
    for i in range(1, len(bins)):
        mask = digitized == i
        if mask.sum() > 10:
            semivariances.append(np.mean(sq_diffs[mask]) / 2)
            bin_centers.append(bins[i])

    sill      = np.percentile(semivariances, 95)
    threshold = 0.95 * sill
    autocorr_length = bin_centers[
        np.argmax(np.array(semivariances) >= threshold)
    ]

    # plot variogram
    plt.figure(figsize=(8, 5))
    plt.scatter(bin_centers, semivariances, alpha=0.6, s=30, color="steelblue")
    plt.axhline(sill, color="red", linestyle="--",
                label=f"Sill: {sill:.2f}")
    plt.axvline(autocorr_length, color="green", linestyle="--",
                label=f"Range: {autocorr_length} px "
                      f"(~{autocorr_length * RESOLUTION_KM:.0f} km)")
    plt.xlabel("Distance (pixels)", fontsize=10)
    plt.ylabel("Semivariance", fontsize=10)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return autocorr_length


# ----- load -----
np.random.seed(42)
hansen = np.load(os.path.join(DATA_DIR, "target_ifl.npy"))
biomes = np.load(os.path.join(DATA_DIR, "biomes_kg_005.npy"))
biomes = np.where(biomes == 5, np.nan, biomes)

# ----- global estimate (year 2000 only) -----
print("=== Global ===")
global_len = estimate_autocorr_length(hansen[0], max_dist_pixels=200, n_samples=3000)
print(f"Global autocorrelation length: {global_len} px "
      f"(~{global_len * RESOLUTION_KM:.0f} km)\n")

# ----- per-biome estimates -----
biome_map = {1: "Tropics", 2: "Arid", 3: "Temperate", 4: "Boreal"}
biome_autocorr = {}

for code, name in biome_map.items():
    print(f"=== {name} ===")
    biome_mask = biomes == code
    data_biome = np.where(biome_mask, hansen[0], np.nan)
    length = estimate_autocorr_length(data_biome, max_dist_pixels=200, n_samples=3000)
    biome_autocorr[name] = length
    print(f"{name}: {length} px (~{length * RESOLUTION_KM:.0f} km)\n")

print("Summary of autocorrelation lengths (pixels):")
for name, length in biome_autocorr.items():
    print(f"  {name}: {length}")
