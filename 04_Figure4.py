"""
Figure 4 – Spatial trends in tree-cover residuals (2000–2023)

Residuals here are the difference between the observed TC and the TC predicted
by the climate-model baseline, so a positive residual means the forest is
denser than the climate alone would suggest, and a negative one means it is
sparser.  Trends in these residuals therefore capture whether the gap between
observed and expected forest cover is growing or shrinking over time.

Four-panel layout:
  a  Global map of the Theil-Sen trend in mean residuals (δRES_avg, %TC yr⁻¹)
  b  Global map of the Theil-Sen trend in residual standard deviation
     (δRES_std, %TC yr⁻¹), with an asymmetric colour scale because increases
     in variability dominate
  c  Grouped bar chart of δRES_avg and δRES_std by biome, with 95 % CI
  d  Pie chart of forest area broken down by biome (latitude-weighted, so
     the slices represent actual ground area rather than pixel counts)

Both maps use a 5×5 moving-window smooth for visual clarity; the statistics
were computed on the unsmoothed grids.

Data required
-------------
global/mk_Residuals_MeanSTD_global_05deg.nc : netCDF with pre-computed
    Mann-Kendall / Theil-Sen results for mean and std of residuals at 0.5°.
global/masked_residuals_global.npy          : float32 array of shape
    (H, W, n_years) containing annual residual TC values at 0.05° resolution.
biomes_kg_05.npy                            : int array (one value per 0.5°
    pixel) with Köppen-Geiger biome class
    (1 = Tropics, 2 = Arid, 3 = Temperate, 4 = Boreal).

Set DATA_DIR and OUTPUT_PATH below before running.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec
from scipy import stats
from scipy.ndimage import generic_filter
import xarray as xr
import cartopy.crs as ccrs
import skimage as sk

# ---------------------------------------------------------------------------
# Paths – adjust these before running
# ---------------------------------------------------------------------------
DATA_DIR    = r"E:\python\TreeCoverDataReview"
OUTPUT_PATH = r"Figure4.pdf"
SAVE_FIGURE = True

# ---------------------------------------------------------------------------
# Load pre-computed trend data
# ---------------------------------------------------------------------------
res = xr.open_dataset(os.path.join(DATA_DIR, "global", "mk_Residuals_MeanSTD_global_05deg.nc"))

# ---------------------------------------------------------------------------
# Load residuals and aggregate to 0.5°
# ---------------------------------------------------------------------------
biomes = np.load(os.path.join(DATA_DIR, "biomes_kg_05.npy"))

# Raw residuals come in (H, W, years) order; transpose to (years, H, W)
# so the time axis is first, consistent with the Hansen array convention.
residuals = np.load(os.path.join(DATA_DIR, "global", "masked_residuals_global.npy"))
residuals = np.transpose(residuals, (2, 0, 1))

# Aggregate 0.05° pixels to 0.5° by summarising each 10×10 block.
# Both the spatial mean and std are needed — mean captures the direction of
# the disequilibrium signal, std captures how heterogeneous it is locally.
residuals_05deg_mean = np.stack([
    sk.measure.block_reduce(residuals[i], block_size=(10, 10), func=np.nanmean)
    for i in range(residuals.shape[0])
])
residuals_05deg_std = np.stack([
    sk.measure.block_reduce(residuals[i], block_size=(10, 10), func=np.nanstd)
    for i in range(residuals.shape[0])
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

def significant_dots(dataset, step, stat=None):
    """
    Pull trend and p-value grids from a Mann-Kendall netCDF and return both
    the full-resolution arrays and a thinned version for stippling.

    'stat' selects which variable pair to read: None → ts_slope / p_value
    (the mean residual trend), 'std' → ts_slope_std / p_value_std.

    'step' controls the stippling density; 6 gives one dot per 6×6-pixel
    block, which avoids overplotting at the target print size.
    """
    if stat is not None:
        trend = np.flipud(dataset[f'ts_slope_{stat}'].values)
        pval  = np.flipud(dataset[f'p_value_{stat}'].values)
    else:
        trend = np.flipud(dataset['ts_slope'].values)
        pval  = np.flipud(dataset['p_value'].values)

    lat = dataset['lat'].values
    lon = dataset['lon'].values

    if lat[0] > lat[-1]:
        trend = trend[::-1, :]
        pval  = pval[::-1, :]
        lat   = lat[::-1]

    lon2d,     lat2d     = np.meshgrid(lon,         lat)
    lon2d_sub, lat2d_sub = np.meshgrid(lon[::step], lat[::step])

    return (lon2d, lat2d, trend, pval,
            lon2d_sub, lat2d_sub, trend[::step, ::step], pval[::step, ::step])


def significant_proportion(pval, alpha=0.05):
    """Fraction of valid grid cells with a trend significant at level alpha."""
    valid = np.isfinite(pval)
    if not valid.any():
        return np.nan
    return ((pval < alpha) & valid).sum() / valid.sum()


def smooth_2d(arr, size=5):
    """
    NaN-aware spatial smoothing with a moving-window mean.

    Used only for display — the statistics were computed on the raw grid.
    Pixels that are NaN in the input stay NaN so ocean / non-forest areas
    don't contaminate adjacent land pixels.
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
    Compute the Theil-Sen slope of the spatial-mean time series for each
    biome (and globally).  Error bars represent the 95 % confidence interval
    of the slope estimate.
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
            "biome":       biome_name,
            "slope":       result.slope,
            "low_slope":   result.low_slope,
            "high_slope":  result.high_slope,
            "error_lower": result.slope - result.low_slope,
            "error_upper": result.high_slope - result.slope,
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
plt.rcParams['font.family']    = 'Arial'
plt.rcParams['font.size']      = 16
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['pdf.fonttype']   = 42

FONT_SIZE  = 18
LABEL_SIZE = 15
TICK_SIZE  = 12

fig = plt.figure(figsize=(18, 11), dpi=400)
gs  = GridSpec(2, 5, figure=fig,
               width_ratios=[1, 1, 1, 1, 1],
               height_ratios=[1, 1])

ax1 = fig.add_subplot(gs[0, :3], projection=ccrs.Robinson())
ax2 = fig.add_subplot(gs[0, 3:])
ax3 = fig.add_subplot(gs[1, :3], projection=ccrs.Robinson())
ax4 = fig.add_subplot(gs[1, 3:])

# ── Panel a: trend in mean residuals ─────────────────────────────────────
lon2d, lat2d, trend, pval, *_ = significant_dots(res, step=6, stat=None)
sig_prop_avg = significant_proportion(pval)

ax1.set_global()
ax1.set_extent([-180, 180, -60, 75], crs=ccrs.PlateCarree())

pcm1 = ax1.pcolormesh(lon2d, lat2d, smooth_2d(trend),
                      cmap='BrBG', transform=ccrs.PlateCarree(),
                      shading='auto', vmin=-0.2, vmax=0.2)
ax1.coastlines(linewidth=0.5)
setup_gl(ax1, TICK_SIZE)

cbar1 = plt.colorbar(pcm1, ax=ax1, orientation='horizontal', shrink=0.5, pad=0.1)
cbar1.set_label(r'$\delta$RES$_{\mathrm{avg}}$ (%TC year$^{-1}$)', fontsize=LABEL_SIZE)
cbar1.ax.tick_params(labelsize=TICK_SIZE)

fig.text(0.03, 0.97, 'a', fontsize=FONT_SIZE, fontweight='bold', va='top', ha='left')

# ── Panel c: biome bar chart ───────────────────────────────────────────────
df_avg = dataframe_trends(residuals_05deg_mean, biomes_dict)
df_std = dataframe_trends(residuals_05deg_std,  biomes_dict)

x     = np.arange(len(df_avg))
width = 0.35

ax2.bar(x - width / 2, df_avg["slope"], width, color='darkgreen', alpha=0.7,
        yerr=[df_avg["error_lower"], df_avg["error_upper"]],
        capsize=5, error_kw={'linewidth': 1.5, 'ecolor': 'black'},
        label=r'$\delta$RES$_{\mathrm{avg}}$')
ax2.bar(x + width / 2, df_std["slope"], width, color='darkorange', alpha=0.7,
        yerr=[df_std["error_lower"], df_std["error_upper"]],
        capsize=5, error_kw={'linewidth': 1.5, 'ecolor': 'black'},
        label=r'$\delta$RES$_{\mathrm{std}}$')

ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax2.set_xticks(x)
ax2.set_xticklabels(df_avg.index, fontsize=LABEL_SIZE, rotation=25, ha='right')
ax2.set_ylabel(r'Theil-Sen Slope (%TC year$^{-1}$)', fontsize=LABEL_SIZE)
ax2.tick_params(axis='y', labelsize=TICK_SIZE)
ax2.grid(axis='y', alpha=0.3)
ax2.legend(fontsize=TICK_SIZE, loc='lower center', framealpha=0.9, ncol=2)

fig.text(0.56, 0.97, 'c', fontsize=FONT_SIZE, fontweight='bold', va='top', ha='left')

# ── Panel b: trend in residual standard deviation ─────────────────────────
lon2d, lat2d, trend, pval, *_ = significant_dots(res, step=6, stat="std")
sig_prop_std = significant_proportion(pval)

ax3.set_global()
ax3.set_extent([-180, 180, -60, 75], crs=ccrs.PlateCarree())

# Increases in residual variability are the dominant signal, so the colour
# scale is asymmetric — the warm side runs to +0.2 while the cool side only
# needs to reach −0.05 to capture the full range.
pcm3 = ax3.pcolormesh(lon2d, lat2d, smooth_2d(trend),
                      cmap='seismic', transform=ccrs.PlateCarree(),
                      norm=TwoSlopeNorm(vmin=-0.05, vcenter=0, vmax=0.2),
                      shading='auto')
ax3.coastlines(linewidth=0.5)
setup_gl(ax3, TICK_SIZE)

cbar3 = plt.colorbar(pcm3, ax=ax3, orientation='horizontal', shrink=0.5, pad=0.1)
cbar3.set_label(r'$\delta$RES$_{\mathrm{std}}$ (%TC year$^{-1}$)', fontsize=LABEL_SIZE)
cbar3.ax.tick_params(labelsize=TICK_SIZE)

fig.text(0.03, 0.49, 'b', fontsize=FONT_SIZE, fontweight='bold', va='top', ha='left')

# ── Panel d: pie chart of forest area by biome (latitude-weighted) ─────────
# A pixel at latitude φ covers less area than one at the equator by a factor
# of cos(φ), so we weight each grid cell by cos(lat) before summing.  This
# gives a fair representation of actual ground area rather than pixel counts.
forest_mask = ~np.all(np.isnan(residuals_05deg_mean), axis=0)

n_lat, n_lon    = biomes.shape
lats_biome      = np.linspace(89.75, -89.75, n_lat)
lat_weights_1d  = np.cos(np.deg2rad(lats_biome))
lat_w2d         = np.tile(lat_weights_1d[:, np.newaxis], (1, n_lon))

biome_labels = ['Tropics', 'Arid', 'Temperate', 'Boreal']
biome_codes  = [1, 2, 3, 4]
biome_colors = ['#2d8a4e', '#c9a227', '#4e7bbf', '#6b3a7d']

areas = [np.sum(lat_w2d[(biomes == code) & forest_mask]) for code in biome_codes]
total = sum(areas)
pcts  = [100.0 * a / total for a in areas]

wedges, _ = ax4.pie(
    areas,
    labels=None,
    colors=biome_colors,
    autopct=None,
    startangle=90,
    wedgeprops={'linewidth': 0.8, 'edgecolor': 'white'},
)

# Place labels manually — automatic positioning overlaps for small wedges.
_r_tip    = 1.05
_label_cfg = {
    'Tropics':   dict(scale=1.30, dx= 0.00, dy= 0.00),
    'Arid':      dict(scale=1.25, dx=-0.42, dy= 0.00),
    'Temperate': dict(scale=1.25, dx= 0.42, dy= 0.00),
    'Boreal':    dict(scale=1.30, dx= 0.00, dy= 0.00),
}

for wedge, label, pct in zip(wedges, biome_labels, pcts):
    ang = (wedge.theta2 + wedge.theta1) / 2.0
    cx  = np.cos(np.deg2rad(ang))
    cy  = np.sin(np.deg2rad(ang))
    cfg = _label_cfg[label]
    tx  = cfg['scale'] * cx + cfg['dx']
    ty  = cfg['scale'] * cy + cfg['dy']
    ax4.annotate(
        f'{label}\n{pct:.1f}%',
        xy=(_r_tip * cx, _r_tip * cy),
        xytext=(tx, ty),
        fontsize=TICK_SIZE + 3,
        ha='left' if tx >= 0 else 'right',
        va='center',
        arrowprops=dict(arrowstyle='-', color='dimgray', lw=0.8,
                        connectionstyle='arc3,rad=0.0'),
    )

ax4.set_title('Forest area by biome\n(latitude-weighted)',
              fontsize=LABEL_SIZE + 5, pad=6)
fig.text(0.56, 0.49, 'd', fontsize=FONT_SIZE, fontweight='bold', va='top', ha='left')

# ---------------------------------------------------------------------------
# Save / show
# ---------------------------------------------------------------------------
plt.tight_layout()

if SAVE_FIGURE:
    fig.savefig(OUTPUT_PATH, bbox_inches='tight')
    print(f"Figure saved to {OUTPUT_PATH}")

plt.show()

print(f"\nSignificant pixels (p < 0.05):")
print(f"  δRES_avg : {sig_prop_avg:.1%}")
print(f"  δRES_std : {sig_prop_std:.1%}")
