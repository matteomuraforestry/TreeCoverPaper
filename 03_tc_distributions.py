"""
03_tc_distributions.py

Analyse and visualise the climatological tree cover distribution from
Hansen data, globally and per biome (Tropics / Arid / Temperate / Boreal).

Applies an arcsine-square-root transformation (arcsin(sqrt(TC/100))) to
stabilise variance before plotting histograms.

Inputs:
    target_ifl.npy      — Hansen TC [years x lat x lon], TC ≥ 10 %
    biomes_kg_005.npy   — Koppen-Geiger biome map at 0.05°
Outputs:
    Figures displayed on screen (extend as needed to save to file).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

DATA_DIR = r""
TC_THRESHOLD = 10

# ----- load data -----
hansen = np.load(os.path.join(DATA_DIR, "target_ifl.npy"))
hansen = np.where(hansen < TC_THRESHOLD, np.nan, hansen)

biomes = np.load(os.path.join(DATA_DIR, "biomes_kg_005.npy"))

# ----- climatological mean and arcsin transform -----
clim_hansen = np.nanmean(hansen, axis=0)
clim_transformed = np.arcsin(np.sqrt(clim_hansen / 100))

print(f"Climatological TC — mean: {np.nanmean(clim_hansen):.2f}%, "
      f"std: {np.nanstd(clim_hansen):.2f}%")

# x-axis ticks mapped back to original percentage scale
labels = np.arange(10, 101, 10)
ticks = np.arcsin(np.sqrt(labels / 100))

# ----- biome masks -----
biome_masks = {
    "Tropics":   biomes == 1,
    "Arid":      biomes == 2,
    "Temperate": biomes == 3,
    "Boreal":    biomes == 4,
}

# ----- area statistics per biome -----
n_lat, n_lon = clim_transformed.shape
lats = np.linspace(-90 + 0.025, 90 - 0.025, n_lat)
lat_grid = np.repeat(lats[:, np.newaxis], n_lon, axis=1)

# pixel area shrinks toward poles
pixel_area_km2 = (0.05 * 111.32) ** 2 * np.cos(np.radians(lat_grid))

total_valid = np.sum(~np.isnan(clim_transformed))
total_area  = np.nansum(np.where(~np.isnan(clim_transformed), pixel_area_km2, np.nan))

print(f"\nTotal valid pixels: {total_valid:,}")
print(f"Total forest area:  {total_area / 1e6:.2f} M km²")
print("-" * 60)

for biome_name, mask in biome_masks.items():
    biome_data = np.where(mask, clim_transformed, np.nan)
    n_pix = np.sum(~np.isnan(biome_data))
    area = np.nansum(np.where(~np.isnan(biome_data), pixel_area_km2, np.nan))
    print(f"{biome_name}: {n_pix:,} pixels  |  {area / 1e6:.2f} M km²  "
          f"({100 * area / total_area:.1f}% of total)")

# ----- climatological distribution plot -----
fig = plt.figure(figsize=(16, 12), dpi=300)
gs = GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.3)

ax_global    = fig.add_subplot(gs[0, 1:3])
ax_tropics   = fig.add_subplot(gs[1, 0:2])
ax_arid      = fig.add_subplot(gs[1, 2:4])
ax_temperate = fig.add_subplot(gs[2, 0:2])
ax_boreal    = fig.add_subplot(gs[2, 2:4])
biome_axes   = [ax_tropics, ax_arid, ax_temperate, ax_boreal]

# global panel
sns.histplot(clim_transformed.ravel(), bins=30, alpha=0.5,
             kde=False, color="#949BA7", ax=ax_global)
ax_global.set_xticks(ticks)
ax_global.set_xticklabels(labels, fontsize=11)
ax_global.tick_params(axis="y", labelsize=11)
ax_global.set_xlabel("TC (%)", fontsize=11)
ax_global.set_ylabel("Frequency", fontsize=11)
ax_global.set_title("Global", fontsize=12, fontweight="bold")
ax_global.grid(axis="y", alpha=0.3, linestyle="--")

# biome panels
axis_config = {
    "Tropics":   (False, True),
    "Arid":      (False, False),
    "Temperate": (True,  True),
    "Boreal":    (True,  False),
}

for idx, (biome_name, mask) in enumerate(biome_masks.items()):
    biome_data = np.where(mask, clim_transformed, np.nan)
    valid_data = biome_data[~np.isnan(biome_data)]
    ax = biome_axes[idx]
    show_xlabel, show_ylabel = axis_config[biome_name]

    sns.histplot(valid_data, bins=30, alpha=0.5,
                 kde=False, ax=ax, color="#949BA7")
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=11)
    ax.tick_params(axis="y", labelsize=11)
    ax.set_xlabel("TC (%)" if show_xlabel else "", fontsize=11)
    ax.set_ylabel("Frequency" if show_ylabel else "", fontsize=11)
    ax.set_title(biome_name, fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--")

plt.suptitle("Climatological TC Distribution (2000-2023)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

# ----- year-to-year anomalies -----
avg_yearly = np.nanmean(hansen.reshape(hansen.shape[0], -1), axis=1)
anomaly    = avg_yearly - avg_yearly[0]
years      = np.arange(2000, 2024)

fig, ax = plt.subplots(figsize=(14, 5))
colors = ["#2e7d32" if v >= 0 else "#c62828" for v in anomaly]
ax.bar(years, anomaly, color=colors, alpha=0.7, edgecolor="black", linewidth=0.5)
ax.axhline(0, color="black", linewidth=1)
ax.set_xlabel("Year", fontsize=11)
ax.set_ylabel("TC Anomaly (%)", fontsize=11)
ax.set_title("Annual TC Anomaly Relative to 2000", fontsize=12, fontweight="bold")
ax.set_xticks(years)
ax.set_xticklabels(years, rotation=45, ha="right")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()
