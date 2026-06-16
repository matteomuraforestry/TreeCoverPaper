"""
08_trend_maps.py

Plot global maps of Mann-Kendall / Theil-Sen trend statistics at 0.5°
resolution on a Robinson projection.

For each variable (TC mean, TC std, TC CV):
  (a) Theil-Sen slope map with dots marking significant pixels (p < 0.05)
  (b) Kendall's tau map

Inputs:
    mk_Hansen_TCgt10IFL_05deg.nc     — trend results for TC mean
    mk_Hansen_STD_TCgt10IFL_05deg.nc — trend results for TC std
    mk_Hansen_CV_TCgt10IFL_05deg.nc  — trend results for TC CV
Outputs:
    Figures displayed on screen.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs

DATA_DIR = r""

# ----- load NetCDF trend files -----
tc  = xr.open_dataset(os.path.join(DATA_DIR, "mk_Hansen_TCgt10IFL_05deg.nc"))
std = xr.open_dataset(os.path.join(DATA_DIR, "mk_Hansen_STD_TCgt10IFL_05deg.nc"))
cv  = xr.open_dataset(os.path.join(DATA_DIR, "mk_Hansen_CV_TCgt10IFL_05deg.nc"))


def prepare_arrays(ds, flip=True):
    """Extract slope, p-value, tau and ensure ascending latitudes."""
    slope = ds["ts_slope"].values
    pval  = ds["p_value"].values
    tau   = ds["kendall_tau"].values
    lat   = ds["lat"].values
    lon   = ds["lon"].values

    if lat[0] > lat[-1]:   # flip if latitudes are descending
        slope = slope[::-1, :]
        pval  = pval[::-1, :]
        tau   = tau[::-1, :]
        lat   = lat[::-1]

    return slope, pval, tau, lat, lon


def subsample_significance(lat, lon, pval, step=3):
    """Downsample grid for significance dot overlay."""
    lat_sub  = lat[::step]
    lon_sub  = lon[::step]
    pval_sub = pval[::step, ::step]
    lon2d_sub, lat2d_sub = np.meshgrid(lon_sub, lat_sub)
    return lon2d_sub, lat2d_sub, pval_sub


def plot_slope_map(slope, pval, lat, lon, cmap, vmin, vmax,
                   cbar_label, title, sig_step=3):
    """Robinson projection map of Theil-Sen slopes with significance dots."""
    lon2d, lat2d = np.meshgrid(lon, lat)
    lon_sub, lat_sub, pval_sub = subsample_significance(lat, lon, pval, sig_step)

    fig = plt.figure(figsize=(14, 8))
    ax  = plt.axes(projection=ccrs.Robinson())
    ax.set_global()
    ax.set_extent([-180, 180, -60, 75], crs=ccrs.PlateCarree())

    pcm = ax.pcolormesh(lon2d, lat2d, slope,
                        cmap=cmap, transform=ccrs.PlateCarree(),
                        shading="auto", vmin=vmin, vmax=vmax)

    sig = pval_sub < 0.05
    ax.plot(lon_sub[sig], lat_sub[sig], "k.",
            markersize=0.5, transform=ccrs.PlateCarree(), zorder=10)

    ax.coastlines(linewidth=0.5)
    gl = ax.gridlines(draw_labels=True, linewidth=0.5,
                      color="gray", alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = plt.FixedLocator(np.arange(-180, 181, 60))
    gl.ylocator = plt.FixedLocator(np.arange(-90, 91, 30))

    plt.colorbar(pcm, ax=ax, orientation="vertical",
                 label=cbar_label, shrink=0.6, pad=0.05)
    ax.set_title(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_tau_map(tau, lat, lon, cmap, title):
    """Robinson projection map of Kendall's tau."""
    lon2d, lat2d = np.meshgrid(lon, lat)

    fig = plt.figure(figsize=(14, 8))
    ax  = plt.axes(projection=ccrs.Robinson())
    ax.set_global()
    ax.set_extent([-180, 180, -60, 75], crs=ccrs.PlateCarree())

    pcm = ax.pcolormesh(lon2d, lat2d, tau,
                        cmap=cmap, transform=ccrs.PlateCarree(),
                        shading="auto", vmin=-1, vmax=1)

    ax.coastlines(linewidth=0.5)
    gl = ax.gridlines(draw_labels=True, linewidth=0.5,
                      color="gray", alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = plt.FixedLocator(np.arange(-180, 181, 60))
    gl.ylocator = plt.FixedLocator(np.arange(-90, 91, 30))

    plt.colorbar(pcm, ax=ax, orientation="vertical",
                 label="Kendall's τ", shrink=0.6, pad=0.05)
    ax.set_title(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ----- TC mean -----
slope_tc, pval_tc, tau_tc, lat_tc, lon_tc = prepare_arrays(tc)

plot_slope_map(slope_tc, pval_tc, lat_tc, lon_tc,
               cmap="BrBG", vmin=-0.35, vmax=0.35,
               cbar_label="Slope (%TC/year)",
               title="TC Mean Trend (2000-2023) — Hansen 0.5°")

plot_tau_map(tau_tc, lat_tc, lon_tc,
             cmap="BrBG",
             title="TC Mean Trend Strength (Kendall's τ) — Hansen 0.5°")

# ----- TC standard deviation -----
slope_std, pval_std, tau_std, lat_std, lon_std = prepare_arrays(std)

plot_slope_map(slope_std, pval_std, lat_std, lon_std,
               cmap="BrBG_r", vmin=-0.35, vmax=0.35,
               cbar_label="Slope (%TC/year)",
               title="TC STD Trend (2000-2023) — Hansen 0.5°")

plot_tau_map(tau_std, lat_std, lon_std,
             cmap="BrBG_r",
             title="TC STD Trend Strength (Kendall's τ) — Hansen 0.5°")

# ----- TC coefficient of variation -----
slope_cv, pval_cv, tau_cv, lat_cv, lon_cv = prepare_arrays(cv)

plot_slope_map(slope_cv, pval_cv, lat_cv, lon_cv,
               cmap="BrBG_r", vmin=-0.005, vmax=0.005,
               cbar_label="Slope (CV/year)",
               title="TC CV Trend (2000-2023) — Hansen 0.5°")

plot_tau_map(tau_cv, lat_cv, lon_cv,
             cmap="BrBG_r",
             title="TC CV Trend Strength (Kendall's τ) — Hansen 0.5°")
