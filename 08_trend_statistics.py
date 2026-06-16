"""
07_trend_statistics.py

Compute Mann-Kendall trends (with 95% confidence intervals) for tree cover
mean, standard deviation, and coefficient of variation at 0.5° resolution.

Confidence intervals account for spatial autocorrelation by reducing the
effective sample size: n_eff = n_actual / autocorr_length (at 0.5°, the
biome-specific lengths in pixels are: Tropics 12, Arid 11, Temperate 18,
Boreal 14).

Inputs:
    hansen_gt10_IFL_05deg_mean_std_cv.nc — TC mean/std/CV at 0.5°
    biomes_kg_05.npy                     — biome map at 0.5°
Outputs:
    Printed Mann-Kendall results per biome.
    Time-series plots with CI bands.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
import pymannkendall as mk

DATA_DIR = r""

# autocorrelation lengths in pixels (0.5° grid) from semivariogram analysis
AUTOCORR = {"Tropics": 12, "Arid": 11, "Temperate": 18, "Boreal": 14}

# ----- load -----
data  = xr.open_dataset(os.path.join(DATA_DIR, "hansen_gt10_IFL_05deg_mean_std_cv.nc"))
tc    = data["TC"].values    # (years, lat, lon)
std   = data["STD"].values
cv    = data["CV"].values
years = np.arange(2000, 2024)

biomes      = np.load(os.path.join(DATA_DIR, "biomes_kg_05.npy"))
biomes_flat = biomes.flatten()


def compute_CI_spatial_autocorr(array, autocorr_length=11):
    """95% CI that accounts for spatial autocorrelation."""
    n_actual    = np.sum(~np.isnan(array))
    n_effective = n_actual / autocorr_length
    sem = np.nanstd(array, ddof=1) / np.sqrt(n_effective)
    return 1.96 * sem


biome_dict = {
    "Tropics":   {"code": 1, "autocorr": AUTOCORR["Tropics"]},
    "Arid":      {"code": 2, "autocorr": AUTOCORR["Arid"]},
    "Temperate": {"code": 3, "autocorr": AUTOCORR["Temperate"]},
    "Boreal":    {"code": 4, "autocorr": AUTOCORR["Boreal"]},
}

n_years = tc.shape[0]


def plot_biome_trends(variable_data, variable_name, color="darkgreen"):
    """Plot annual mean ± CI for each biome and report Mann-Kendall results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, (biome_name, biome_info) in enumerate(biome_dict.items()):
        ax = axes[i]
        mask = biomes_flat == biome_info["code"]
        biome_data = variable_data.reshape(n_years, -1)[:, mask]

        avg = np.nanmean(biome_data, axis=1)
        ci  = np.apply_along_axis(
            compute_CI_spatial_autocorr, 1, biome_data,
            autocorr_length=biome_info["autocorr"]
        )

        mk_result = mk.original_test(avg)
        print(f"{biome_name} {variable_name} Mann-Kendall: {mk_result}")

        ax.plot(years, avg, color=color, linewidth=2)
        ax.fill_between(years, avg - ci, avg + ci, alpha=0.2, color=color)
        ax.set_title(biome_name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Year", fontsize=10)
        ax.set_ylabel(f"{variable_name}", fontsize=10)
        ax.grid(alpha=0.3)

    plt.suptitle(f"Annual {variable_name} by Biome (2000-2023)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ----- global trend -----
tc_flat = tc.reshape(n_years, -1)
avg_global_tc = np.nanmean(tc_flat, axis=1)
tc_ci_global  = np.apply_along_axis(compute_CI_spatial_autocorr, 1, tc_flat)

print("Global TC Mann-Kendall:", mk.original_test(avg_global_tc))

fig, ax = plt.subplots(figsize=(10, 5))
sns.lineplot(x=years, y=avg_global_tc, color="darkgreen", ax=ax)
ax.fill_between(years, avg_global_tc - tc_ci_global,
                avg_global_tc + tc_ci_global, alpha=0.2, color="darkgreen")
ax.set_title("Global Mean TC (2000-2023)", fontsize=12, fontweight="bold")
ax.set_xlabel("Year")
ax.set_ylabel("TC (%)")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ----- per-biome trends -----
print("\n--- Tree Cover mean ---")
plot_biome_trends(tc, "TC (%)")

print("\n--- Tree Cover standard deviation ---")
plot_biome_trends(std, "TC STD (%)")

print("\n--- Tree Cover CV ---")
plot_biome_trends(cv, "TC CV")
