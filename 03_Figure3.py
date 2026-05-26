"""
Figure 3 – Spatial trends in mean and variability of tree cover (2000–2023)

Four-panel figure:
  a  Global map of the Theil-Sen trend in mean TC (δTC_avg, %TC yr⁻¹)
  b  Global map of the Theil-Sen trend in TC standard deviation (δTC_std, %TC yr⁻¹)
  c  Grouped bar chart of those same trends broken down by biome, with 95 % CI
  d  Hexbin scatter of δTC_std vs. δTC_avg across all 0.5° grid cells, with a
     Pearson correlation coefficient and regression line

Both maps use a spatially smoothed (5×5 moving window) version of the trend
layer for visual clarity; the statistical tests were run on the unsmoothed data.
The asymmetric colour scales on the maps reflect the fact that TC losses are
typically larger in magnitude than TC gains.

Data required
-------------
mk_Hansen_TCgt10IFL_05deg.nc  : netCDF with pre-computed Mann-Kendall / Theil-Sen
                                 results for mean TC at 0.5° resolution.
mk_Hansen_STD_TCgt10IFL_05deg.nc : same for TC standard deviation.
target_ifl.npy                : float32 array (n_years, H, W) of annual Hansen TC
                                 values at 0.05° resolution (2000–2023).
biomes_kg_05.npy              : int array with one value per 0.5° pixel encoding
                                 the Köppen-Geiger biome class
                                 (1 = Tropics, 2 = Arid, 3 = Temperate, 4 = Boreal).

Set DATA_DIR and OUTPUT_PATH below before running.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec
from scipy import stats
from scipy.stats import pearsonr
from scipy.ndimage import generic_filter
import xarray as xr
import cartopy.crs as ccrs
import seaborn as sns
import skimage as sk

# ---------------------------------------------------------------------------
# Paths – adjust these before running
# ---------------------------------------------------------------------------
DATA_DIR    = r"E:\python\TreeCoverDataReview"
OUTPUT_PATH = r"Figure3.pdf"
SAVE_FIGURE = True

# ---------------------------------------------------------------------------
# Load pre-computed trend data (Mann-Kendall / Theil-Sen, run separately)
# ---------------------------------------------------------------------------
tc  = xr.open_dataset(os.path.join(DATA_DIR, "mk_Hansen_TCgt10IFL_05deg.nc"))
std = xr.open_dataset(os.path.join(DATA_DIR, "mk_Hansen_STD_TCgt10IFL_05deg.nc"))

# ---------------------------------------------------------------------------
# Load raw tree-cover data to compute biome-level trends
# ---------------------------------------------------------------------------
biomes = np.load(os.path.join(DATA_DIR, "biomes_kg_05.npy"))    # 0.5° biome map

hansen = np.load(os.path.join(DATA_DIR, "target_ifl.npy"))      # 0.05° annual TC
TC_THRESHOLD = 10
hansen = np.where(hansen < TC_THRESHOLD, np.nan, hansen)

# Aggregate 0.05° pixels to 0.5° by taking the spatial mean / std over each
# 10×10 block.  This matches the resolution of the trend netCDFs.
hansen_05deg_mean = np.stack([
    sk.measure.block_reduce(hansen[i], block_size=(10, 10), func=np.nanmean)
    for i in range(hansen.shape[0])
])
hansen_05deg_std = np.stack([
    sk.measure.block_reduce(hansen[i], block_size=(10, 10), func=np.nanstd)
    for i in range(hansen.shape[0])
])

biomes_dict = {
    "Global":    None,
    "Tropics":   1,
    "Arid":      2,
    "Temperate": 3,
    "Boreal":    4,
}

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def significant_dots(dataset, step):
    """
    Extract trend and p-value arrays from a Mann-Kendall netCDF dataset and
    return both the full-resolution grids and a subsampled version suitable
    for plotting significance stippling without overplotting.

    'step' controls how aggressively the stippling is thinned — a value of 6
    means we plot one dot per 6×6-pixel block, which looks clean at the
    figure's output resolution.
    """
    trend = dataset['ts_slope'].values
    pval  = dataset['p_value'].values
    lat   = dataset['lat'].values
    lon   = dataset['lon'].values

    # Some datasets store latitude descending; flip so it runs south → north.
    if lat[0] > lat[-1]:
        trend = trend[::-1, :]
        pval  = pval[::-1, :]
        lat   = lat[::-1]

    lat_sub   = lat[::step]
    lon_sub   = lon[::step]
    trend_sub = trend[::step, ::step]
    pval_sub  = pval[::step, ::step]

    lon2d,     lat2d     = np.meshgrid(lon,     lat)
    lon2d_sub, lat2d_sub = np.meshgrid(lon_sub, lat_sub)

    return lon2d, lat2d, trend, pval, lon2d_sub, lat2d_sub, trend_sub, pval_sub


def significant_proportion(pval, alpha=0.05):
    """What fraction of valid pixels have a trend significant at level alpha?"""
    valid   = np.isfinite(pval)
    n_valid = valid.sum()
    if n_valid == 0:
        return np.nan
    return ((pval < alpha) & valid).sum() / n_valid


def smooth_2d(arr, size=5):
    """
    NaN-aware spatial smoothing using a moving window average.

    We use this only for display — the underlying statistics are computed on
    the raw grid.  Pixels that are NaN in the input stay NaN in the output so
    ocean / non-forest areas don't bleed into adjacent land pixels.
    """
    def nan_mean(x):
        center = x[len(x) // 2]
        if np.isnan(center):
            return np.nan
        valid = x[~np.isnan(x)]
        return np.nanmean(valid) if len(valid) > 0 else np.nan

    return generic_filter(arr, nan_mean, size=size, mode='nearest')


def dataframe_trends(data, biome_dict):
    """
    Compute the Theil-Sen slope of the spatial mean time series for each
    biome (and globally).  Returns a DataFrame with the slope and its 95 %
    confidence interval bounds so we can draw error bars.
    """
    rows = []
    for biome_name, biome_code in biome_dict.items():
        if biome_code is None:
            data_biome = data.copy()
        else:
            data_biome = np.where(biomes == biome_code, data, np.nan)

        ts     = np.nanmean(data_biome.reshape(data_biome.shape[0], -1), axis=1)
        result = stats.theilslopes(ts)

        rows.append({
            "biome":        biome_name,
            "slope":        result.slope,
            "low_slope":    result.low_slope,
            "high_slope":   result.high_slope,
            "error_lower":  result.slope - result.low_slope,
            "error_upper":  result.high_slope - result.slope,
        })

    return pd.DataFrame(rows).set_index("biome")


def setup_gl(ax, tick_size):
    """Standard gridline configuration for Cartopy Robinson-projection maps."""
    gl = ax.gridlines(
        draw_labels={'left': True, 'right': False, 'bottom': True, 'top': False},
        linewidth=0.5, color='gray', alpha=0.5, linestyle='--'
    )
    gl.xlabel_style = {'size': tick_size}
    gl.ylabel_style = {'size': tick_size}
    gl.xlocator = plt.FixedLocator(np.arange(-180, 181, 60))
    gl.ylocator = plt.FixedLocator(np.arange(-90,   91, 30))
    return gl


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
plt.rcParams['font.family']     = 'Arial'
plt.rcParams['font.size']       = 16
plt.rcParams['axes.linewidth']  = 1.0
plt.rcParams['pdf.fonttype']    = 42

FONT_SIZE  = 18   # panel letter size
LABEL_SIZE = 15   # axis and colorbar labels
TICK_SIZE  = 12   # tick labels

fig = plt.figure(figsize=(18, 11), dpi=500)
gs  = GridSpec(2, 2, figure=fig,
               width_ratios=[1.5, 1], height_ratios=[1, 1],
               hspace=0.25, wspace=0.30,
               left=0.02, right=0.98, top=0.98, bottom=0.05)

ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.Robinson())
ax2 = fig.add_subplot(gs[1, 0], projection=ccrs.Robinson())
ax3 = fig.add_subplot(gs[0, 1])
ax4 = fig.add_subplot(gs[1, 1])

# ── Panel a: trend in mean TC ─────────────────────────────────────────────
lon2d, lat2d, trend, pval, *_ = significant_dots(tc, step=6)
sig_prop_avg = significant_proportion(pval)

# Smooth purely for display — raw heterogeneity at 0.5° makes the maps hard
# to read at the journal's print size.
trend_smooth = smooth_2d(trend, size=5)

ax1.set_global()
ax1.set_extent([-180, 180, -60, 75], crs=ccrs.PlateCarree())

# Asymmetric norm: losses reach −0.2 but gains rarely exceed +0.05,
# so a symmetric scale would wash out most of the signal.
pcm1 = ax1.pcolormesh(lon2d, lat2d, trend_smooth,
                      cmap='BrBG',
                      norm=TwoSlopeNorm(vmin=-0.2, vcenter=0, vmax=0.05),
                      transform=ccrs.PlateCarree(), shading='auto')
ax1.coastlines(linewidth=0.5)
setup_gl(ax1, TICK_SIZE)

cbar1 = plt.colorbar(pcm1, ax=ax1, orientation='horizontal', shrink=0.5, pad=0.08)
cbar1.set_label(r'$\delta$TC$_{\mathrm{avg}}$ (%TC year$^{-1}$)', fontsize=LABEL_SIZE)
cbar1.ax.tick_params(labelsize=TICK_SIZE)

fig.text(0.01, 0.99, 'a', fontsize=FONT_SIZE, fontweight='bold', va='top', ha='left')

# ── Panel b: trend in TC standard deviation ───────────────────────────────
lon2d, lat2d, trend, pval, *_ = significant_dots(std, step=6)
sig_prop_std = significant_proportion(pval)

trend_smooth = smooth_2d(trend, size=5)

ax2.set_global()
ax2.set_extent([-180, 180, -60, 75], crs=ccrs.PlateCarree())

# The std trend has the opposite skew: increases dominate, so we flip the
# asymmetry relative to panel a.
pcm2 = ax2.pcolormesh(lon2d, lat2d, trend_smooth,
                      cmap='seismic',
                      norm=TwoSlopeNorm(vmin=-0.05, vcenter=0, vmax=0.2),
                      transform=ccrs.PlateCarree(), shading='auto')
ax2.coastlines(linewidth=0.5)
setup_gl(ax2, TICK_SIZE)

cbar2 = plt.colorbar(pcm2, ax=ax2, orientation='horizontal', shrink=0.5, pad=0.08)
cbar2.set_label(r'$\delta$TC$_{\mathrm{std}}$ (%TC year$^{-1}$)', fontsize=LABEL_SIZE)
cbar2.ax.tick_params(labelsize=TICK_SIZE)

fig.text(0.01, 0.50, 'b', fontsize=FONT_SIZE, fontweight='bold', va='top', ha='left')

# ── Panel c: biome-level bar chart (mean and std side by side) ────────────
df_avg = dataframe_trends(hansen_05deg_mean, biomes_dict)
df_std = dataframe_trends(hansen_05deg_std,  biomes_dict)

x     = np.arange(len(df_avg))
width = 0.35

ax3.bar(x - width / 2, df_avg["slope"], width, color='darkgreen', alpha=0.7,
        yerr=[df_avg["error_lower"], df_avg["error_upper"]],
        capsize=5, error_kw={'linewidth': 1.5, 'ecolor': 'black'},
        label=r'$\delta$TC$_{\mathrm{avg}}$')

ax3.bar(x + width / 2, df_std["slope"], width, color='darkorange', alpha=0.7,
        yerr=[df_std["error_lower"], df_std["error_upper"]],
        capsize=5, error_kw={'linewidth': 1.5, 'ecolor': 'black'},
        label=r'$\delta$TC$_{\mathrm{std}}$')

ax3.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax3.set_xticks(x)
ax3.set_xticklabels(df_avg.index, fontsize=LABEL_SIZE, rotation=25, ha='right')
ax3.set_ylabel(r'Theil-Sen Slope (%TC year$^{-1}$)', fontsize=LABEL_SIZE)
ax3.tick_params(axis='y', labelsize=TICK_SIZE)
ax3.grid(axis='y', alpha=0.3)
ax3.legend(fontsize=TICK_SIZE, loc='lower center', framealpha=0.9, ncol=2)

fig.text(0.6, 0.99, 'c', fontsize=FONT_SIZE, fontweight='bold', va='top', ha='left')

# ── Panel d: pixel-level scatter of δTC_std vs. δTC_avg ───────────────────
trend_avg_flat = tc['ts_slope'].values.flatten()
trend_std_flat = std['ts_slope'].values.flatten()

mask            = ~(np.isnan(trend_std_flat) | np.isnan(trend_avg_flat))
trend_std_clean = trend_std_flat[mask]
trend_avg_clean = trend_avg_flat[mask]

corr, p_value = pearsonr(trend_std_clean, trend_avg_clean)

hb = ax4.hexbin(trend_std_flat, trend_avg_flat,
                gridsize=80, cmap='viridis',
                mincnt=1, norm=mcolors.LogNorm(), alpha=0.8)

# Regression line with 95 % CI shading
sns.regplot(x=trend_std_clean, y=trend_avg_clean,
            scatter=False, color='red',
            line_kws={'linewidth': 2}, ci=95, ax=ax4)

ax4.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)

cbar4 = plt.colorbar(hb, ax=ax4, shrink=0.7)
cbar4.set_label('Count (log scale)', fontsize=LABEL_SIZE)
cbar4.ax.tick_params(labelsize=TICK_SIZE)

ax4.set_xlabel(r'$\delta$TC$_{\mathrm{std}}$ (%TC year$^{-1}$)', fontsize=LABEL_SIZE)
ax4.set_ylabel(r'$\delta$TC$_{\mathrm{avg}}$ (%TC year$^{-1}$)', fontsize=LABEL_SIZE)
ax4.tick_params(axis='both', labelsize=TICK_SIZE)

p_text = ('p < 0.001' if p_value < 0.001
          else 'p < 0.05' if p_value < 0.05
          else f'p = {p_value:.3f}')
ax4.text(0.95, 0.05, f'r = {corr:.3f}\n{p_text}',
         transform=ax4.transAxes,
         fontsize=LABEL_SIZE, va='bottom', ha='right',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'))

fig.text(0.6, 0.50, 'd', fontsize=FONT_SIZE, fontweight='bold', va='top', ha='left')

# ---------------------------------------------------------------------------
# Save / show
# ---------------------------------------------------------------------------
if SAVE_FIGURE:
    fig.savefig(OUTPUT_PATH, bbox_inches='tight')
    print(f"Figure saved to {OUTPUT_PATH}")

plt.show()

print(f"\nSignificant pixels (p < 0.05):")
print(f"  δTC_avg : {sig_prop_avg:.1%}")
print(f"  δTC_std : {sig_prop_std:.1%}")
