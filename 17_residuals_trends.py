"""
18_residuals_trends.py

Compute Mann-Kendall / Theil-Sen trends of the residuals at two resolutions:

  0.05°  — pixel-level MK test applied directly to the time series
            → mk_Residuals_{area}_005deg.nc

  0.5°   — residuals aggregated to 0.5° first (10×10 block mean and STD)
            MK test applied to mean and STD time series
            → mk_Residuals_MeanSTD_{area}_05deg.nc

Inputs:
    masked_residuals_{area}.npy  — (lat, lon, years) residual arrays
Outputs:
    NetCDF trend files saved to DATA_DIR/{area}/
"""

import os
import numpy as np
import xarray as xr
import skimage.measure as sk_measure
import pymannkendall as mk
from tqdm import tqdm

DATA_DIR = r""
AREAS    = ["global", "arid", "tropics", "temperate", "boreal"]


def mk_test_safe(x):
    """Pixel-wise MK test; returns [trend_dir, slope, p_value, tau]."""
    x_clean = x[~np.isnan(x)]
    if len(x_clean) < 3 or np.std(x_clean) == 0:
        return np.array([np.nan, np.nan, np.nan, np.nan])
    try:
        res = mk.original_test(x_clean)
        trend_num = 1 if res.trend == "increasing" else (
            -1 if res.trend == "decreasing" else 0)
        return np.array([trend_num, res.slope, res.p, res.Tau])
    except (ValueError, ZeroDivisionError):
        return np.array([np.nan, np.nan, np.nan, np.nan])


def save_nc(ds_dict, lat_arr, lon_arr, out_path, desc):
    """Build an xr.Dataset and write to NetCDF."""
    ds = xr.Dataset(
        {k: (["lat", "lon"], v) for k, v in ds_dict.items()},
        coords={"lat": lat_arr, "lon": lon_arr},
        attrs={"description": desc,
               "units": ("trend: 1=inc/-1=dec/0=no; "
                         "ts_slope: %TC/year; p_value: sig; "
                         "kendall_tau: correlation")},
    )
    ds.to_netcdf(out_path)
    print(f"  Saved {out_path}")


for area in tqdm(AREAS, desc="Areas"):
    print(f"\n{area.upper()}")
    data = np.load(
        os.path.join(DATA_DIR, area, f"masked_residuals_{area}.npy")
    ).transpose(2, 0, 1)   # → (years, lat, lon)
    n_years, n_lat, n_lon = data.shape

    lat = np.linspace(-90, 90, n_lat)
    lon = np.linspace(-180, 180, n_lon)

    # ----- 0.05° trends -----
    print("  Computing MK at 0.05° …")
    data_flat = data.reshape(n_years, -1)
    mk_res    = np.zeros((4, data_flat.shape[1]))
    for i in tqdm(range(data_flat.shape[1]),
                  desc=f"  MK pixels {area}", leave=False):
        mk_res[:, i] = mk_test_safe(data_flat[:, i])

    save_nc(
        {
            "trend":       mk_res[0].reshape(n_lat, n_lon),
            "ts_slope":    mk_res[1].reshape(n_lat, n_lon),
            "p_value":     mk_res[2].reshape(n_lat, n_lon),
            "kendall_tau": mk_res[3].reshape(n_lat, n_lon),
        },
        lat, lon,
        os.path.join(DATA_DIR, area, f"mk_Residuals_{area}_005deg.nc"),
        f"MK trends of residuals at 0.05° — {area}",
    )

    # ----- 0.5° trends (block-aggregated mean and STD) -----
    print("  Aggregating to 0.5° …")
    mean_05 = np.stack([
        sk_measure.block_reduce(data[i], (10, 10), np.nanmean)
        for i in range(n_years)
    ])
    std_05  = np.stack([
        sk_measure.block_reduce(data[i], (10, 10), np.nanstd)
        for i in range(n_years)
    ])

    lat_05 = np.linspace(-90, 90, mean_05.shape[1])
    lon_05 = np.linspace(-180, 180, mean_05.shape[2])

    mk_mean = np.apply_along_axis(
        mk_test_safe, 0, mean_05.reshape(n_years, -1)
    )
    mk_std  = np.apply_along_axis(
        mk_test_safe, 0, std_05.reshape(n_years, -1)
    )

    nl, nl2 = mean_05.shape[1], mean_05.shape[2]
    save_nc(
        {
            "trend":          mk_mean[0].reshape(nl, nl2),
            "ts_slope":       mk_mean[1].reshape(nl, nl2),
            "p_value":        mk_mean[2].reshape(nl, nl2),
            "kendall_tau":    mk_mean[3].reshape(nl, nl2),
            "trend_std":      mk_std[0].reshape(nl, nl2),
            "ts_slope_std":   mk_std[1].reshape(nl, nl2),
            "p_value_std":    mk_std[2].reshape(nl, nl2),
            "kendall_tau_std":mk_std[3].reshape(nl, nl2),
        },
        lat_05, lon_05,
        os.path.join(DATA_DIR, area, f"mk_Residuals_MeanSTD_{area}_05deg.nc"),
        f"MK trends of residual mean and STD at 0.5° — {area}",
    )

print("\nDone.")
