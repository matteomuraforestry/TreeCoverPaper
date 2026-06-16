"""
19_residuals_visualization.py

Visualise the spatial and temporal structure of model residuals:

  (a) Climatological residual map (2000-2023 mean, smoothed 5×5 window)
      with per-biome statistics.

  (b) Time series of mean and STD residuals with 95% CI (spatial
      autocorrelation corrected) — global and per biome.

  (c) Publication figure: 4-panel map + bar chart for residual mean
      and STD Theil-Sen slopes.

  (d) Climatological distribution histograms (global + biomes).

Inputs:
    masked_residuals_global.npy          — global residuals (lat, lon, years)
    mk_Residuals_MeanSTD_global_05deg.nc — trend results
    biomes_kg_005.npy                    — biome map at 0.05°
    biomes_kg_05.npy                     — biome map at 0.5°
Outputs:
    Figures displayed on screen.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import xarray as xr
import cartopy.crs as ccrs
import skimage.measure as sk_measure
import pymannkendall as mk
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import ScalarFormatter
from scipy import stats
from scipy.ndimage import generic_filter

DATA_DIR = r""

AUTOCORR_05 = {"Tropics": 12, "Arid": 11, "Temperate": 18, "Boreal": 14}
YEARS = np.arange(2000, 2024)
CM    = 1 / 2.54

plt.rcParams.update({"font.family": "Arial", "font.size": 9,
                     "pdf.fonttype": 42})

# ----- load global residuals and trend file -----
residuals = np.load(os.path.join(DATA_DIR, "global", "masked_residuals_global.npy"))
# (lat, lon, years)
res_transposed = residuals.transpose(2, 0, 1)  # (years, lat, lon)

res_nc   = xr.open_dataset(
    os.path.join(DATA_DIR, "global", "mk_Residuals_MeanSTD_global_05deg.nc")
)
biomes_05  = np.load(os.path.join(DATA_DIR, "biomes_kg_05.npy"))
biomes_005 = np.load(os.path.join(DATA_DIR, "biomes_kg_005.npy"))

# 0.5° aggregated time series
res_mean_05 = np.stack([
    sk_measure.block_reduce(res_transposed[i], (10, 10), np.nanmean)
    for i in range(res_transposed.shape[0])
])
res_std_05  = np.stack([
    sk_measure.block_reduce(res_transposed[i], (10, 10), np.nanstd)
    for i in range(res_transposed.shape[0])
])


def compute_CI(array, autocorr_length=11):
    """95% CI corrected for spatial autocorrelation."""
    n_eff = np.sum(~np.isnan(array)) / autocorr_length
    return 1.96 * np.nanstd(array, ddof=1) / np.sqrt(n_eff)


# ============================================================
# (a) Climatological residual map
# ============================================================
clim_2d = np.nanmean(residuals, axis=2)
clim_2d_clean = np.where(clim_2d == 0, np.nan, clim_2d)

# NaN-aware 5×5 smoothing
def smooth_nan(arr, size=5):
    def _nan_mean(x):
        c = x[len(x) // 2]
        return np.nan if np.isnan(c) else np.nanmean(x[~np.isnan(x)])
    return generic_filter(arr, _nan_mean, size=size, mode="nearest")

clim_smoothed = smooth_nan(clim_2d_clean)

fig = plt.figure(figsize=(14, 8), dpi=300)
ax  = plt.axes(projection=ccrs.Robinson())
ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
ax.coastlines(linewidth=0.5)

im = ax.imshow(clim_smoothed, extent=[-180, 180, -90, 90],
               transform=ccrs.PlateCarree(), origin="upper",
               interpolation="nearest", cmap="BrBG", vmin=-0.5, vmax=0.5)
cbar = plt.colorbar(im, ax=ax, orientation="horizontal",
                    pad=0.05, shrink=0.6, label="Mean Residuals (2000-2023) [%TC]")
gl = ax.gridlines(draw_labels=False, linewidth=0.3, color="gray", alpha=0.5)
plt.title("Climatological Residuals (2000-2023)", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.show()

# print per-biome climatological means
biomes_dict = {"Tropics": 1, "Arid": 2, "Temperate": 3, "Boreal": 4}
for name, code in biomes_dict.items():
    vals = clim_2d_clean[biomes_005 == code]
    print(f"{name}: mean residual = {np.nanmean(vals):.4f} %TC")

# ============================================================
# (b) Time series of global mean and STD residuals
# ============================================================
avg_res_mean = np.nanmean(res_mean_05.reshape(len(YEARS), -1), axis=1)
ci_res_mean  = np.apply_along_axis(
    compute_CI, 1, res_mean_05.reshape(len(YEARS), -1)
)

print("\nGlobal residual mean MK:", mk.original_test(avg_res_mean))

fig, ax = plt.subplots(figsize=(10, 5))
sns.lineplot(x=YEARS, y=avg_res_mean, color="darkgreen", ax=ax)
ax.fill_between(YEARS, avg_res_mean - ci_res_mean,
                avg_res_mean + ci_res_mean, alpha=0.2, color="darkgreen")
ax.set_title("Global Residual Mean (2000-2023)", fontsize=12, fontweight="bold")
ax.set_xlabel("Year")
ax.set_ylabel("Mean Residual (%TC)")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

avg_res_std = np.nanmean(res_std_05.reshape(len(YEARS), -1), axis=1)
ci_res_std  = np.apply_along_axis(
    compute_CI, 1, res_std_05.reshape(len(YEARS), -1)
)

print("Global residual STD MK:", mk.original_test(avg_res_std))

fig, ax = plt.subplots(figsize=(10, 5))
sns.lineplot(x=YEARS, y=avg_res_std, color="darkgreen", ax=ax)
ax.fill_between(YEARS, avg_res_std - ci_res_std,
                avg_res_std + ci_res_std, alpha=0.2, color="darkgreen")
ax.set_title("Global Residual STD (2000-2023)", fontsize=12, fontweight="bold")
ax.set_xlabel("Year")
ax.set_ylabel("Residual STD (%TC)")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================
# (c) Publication figure: trend maps + bar charts
# ============================================================

def significant_dots(ds, step, stat=None):
    """Extract slope, p-value and meshgrids for significance dot overlay."""
    sfx = f"_{stat}" if stat else ""
    slope = np.flipud(ds[f"ts_slope{sfx}"].values)
    pval  = np.flipud(ds[f"p_value{sfx}"].values)
    lat   = ds["lat"].values
    lon   = ds["lon"].values
    if lat[0] > lat[-1]:
        slope = slope[::-1]; pval = pval[::-1]; lat = lat[::-1]
    lon2d, lat2d = np.meshgrid(lon, lat)
    lon_sub, lat_sub = np.meshgrid(lon[::step], lat[::step])
    pval_sub = pval[::step, ::step]
    return lon2d, lat2d, slope, pval, lon_sub, lat_sub, pval_sub


biomes_dict_full = {"Global": None, "Tropics": 1, "Arid": 2,
                    "Temperate": 3, "Boreal": 4}

def dataframe_trends(data, biomes, biomes_dict):
    """Theil-Sen slopes per biome from a (years, lat, lon) array."""
    import pandas as pd
    rows = []
    for name, code in biomes_dict.items():
        arr = data.copy() if code is None else np.where(biomes == code, data, np.nan)
        ts  = np.nanmean(arr.reshape(arr.shape[0], -1), axis=1)
        res = stats.theilslopes(ts)
        rows.append({"biome": name, "slope": res.slope,
                     "low_slope": res.low_slope, "high_slope": res.high_slope,
                     "error_lower": res.slope - res.low_slope,
                     "error_upper": res.high_slope - res.slope})
    import pandas as pd
    return pd.DataFrame(rows).set_index("biome")

df_mean = dataframe_trends(res_mean_05, biomes_05, biomes_dict_full)
df_std  = dataframe_trends(res_std_05,  biomes_05, biomes_dict_full)

fig = plt.figure(figsize=(25 * CM, 18 * CM), dpi=400)
gs  = GridSpec(2, 5, figure=fig, width_ratios=[1, 1, 1, 1, 1])

ax1 = fig.add_subplot(gs[0, :3], projection=ccrs.Robinson())
ax2 = fig.add_subplot(gs[0, 3:])
ax3 = fig.add_subplot(gs[1, :3], projection=ccrs.Robinson())
ax4 = fig.add_subplot(gs[1, 3:])

font_size = 9

for ax, stat, cmap, ax_bar, df, label_top, label_bot in [
    (ax1, None,  "BrBG",   ax2, df_mean, "a", "b"),
    (ax3, "std", "BrBG_r", ax4, df_std,  "c", "d"),
]:
    lon2d, lat2d, slope, pval, lon_sub, lat_sub, pval_sub = \
        significant_dots(res_nc, step=6, stat=stat)

    ax.set_global()
    ax.set_extent([-180, 180, -60, 75], crs=ccrs.PlateCarree())
    pcm = ax.pcolormesh(lon2d, lat2d, slope, cmap=cmap,
                        transform=ccrs.PlateCarree(), shading="auto",
                        vmin=-0.2, vmax=0.2)
    sig = pval_sub < 0.05
    ax.plot(lon_sub[sig], lat_sub[sig], "k.", markersize=0.5,
            transform=ccrs.PlateCarree(), zorder=10)
    ax.coastlines(linewidth=0.5)
    gl = ax.gridlines(draw_labels=True, linewidth=0.5,
                      color="gray", alpha=0.5, linestyle="--")
    gl.top_labels = False; gl.right_labels = False
    gl.xlocator = plt.FixedLocator(np.arange(-180, 181, 60))
    gl.ylocator = plt.FixedLocator(np.arange(-90, 91, 30))
    plt.colorbar(pcm, ax=ax, orientation="horizontal",
                 label="Theil-Sen Slope (%TC/year)", shrink=0.5, pad=0.1)

    x = np.arange(len(df))
    ax_bar.bar(x, df["slope"], width=0.5, color="grey", alpha=0.7,
               yerr=[df["error_lower"], df["error_upper"]],
               capsize=8, error_kw={"linewidth": 2, "ecolor": "black"})
    ax_bar.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(df.index, fontsize=font_size - 2,
                           rotation=45, ha="right")
    ax_bar.set_ylabel("Theil-Sen Slope (%TC/year)", fontsize=font_size - 2)
    ax_bar.grid(axis="y", alpha=0.3)

fig.text(0.03, 0.97, "a", fontsize=font_size + 2, fontweight="bold",
         va="top", ha="left")
fig.text(0.59, 0.97, "b", fontsize=font_size + 2, fontweight="bold",
         va="top", ha="left")
fig.text(0.03, 0.49, "c", fontsize=font_size + 2, fontweight="bold",
         va="top", ha="left")
fig.text(0.58, 0.49, "d", fontsize=font_size + 2, fontweight="bold",
         va="top", ha="left")

plt.tight_layout()
plt.show()

# ============================================================
# (d) Climatological distribution histograms
# ============================================================
areas = ["global", "tropics", "arid", "temperate", "boreal"]
clim_flat_list = []

for area in areas:
    data_a = np.load(os.path.join(DATA_DIR, area, f"masked_residuals_{area}.npy"))
    clim_flat_list.append(np.nanmean(data_a, axis=2).flatten())

fig = plt.figure(figsize=(25 * CM, 18 * CM), dpi=400)
gs  = GridSpec(3, 4, figure=fig, hspace=0.5, wspace=0.5)
axes_dict = {
    "global":    fig.add_subplot(gs[0, 1:3]),
    "tropics":   fig.add_subplot(gs[1, :2]),
    "arid":      fig.add_subplot(gs[1, 2:]),
    "temperate": fig.add_subplot(gs[2, :2]),
    "boreal":    fig.add_subplot(gs[2, 2:]),
}

for i, area in enumerate(areas):
    ax = axes_dict[area]
    sns.histplot(clim_flat_list[i], bins=50, ax=ax, color="grey",
                 edgecolor="black")
    ax.set_title(area.capitalize(), fontsize=10, fontweight="bold")
    ax.set_xlim(-30, 30)

    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((-2, 2))
    ax.yaxis.set_major_formatter(fmt)

    is_left  = area in ("global", "tropics", "temperate")
    is_bottom = area in ("temperate", "boreal", "global")
    ax.set_xlabel("Residuals (%TC)" if is_bottom else "", fontsize=8)
    ax.set_ylabel("Count" if is_left else "", fontsize=8)

plt.tight_layout()
plt.show()
