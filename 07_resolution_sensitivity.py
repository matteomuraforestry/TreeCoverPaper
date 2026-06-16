"""
07_resolution_sensitivity.py

Test sensitivity of Mann-Kendall / Theil-Sen trend estimates to spatial
resolution by aggregating the 0.05° Hansen dataset to 0.25°, 0.5°, and 1°
before computing trends.

TC mean, STD, and CV are computed at each resolution and the resulting
trend maps are saved as NetCDF files.  The 0.5° TC/STD/CV time series is
also saved as a single combined NetCDF used by downstream analysis scripts.

Inputs:
    target_ifl.npy  — Hansen TC [years x lat x lon], TC ≥ 10 %
Outputs:
    mk_Hansen_TCgt10IFL_025deg.nc      mk_Hansen_TCgt10IFL_05deg.nc
    mk_Hansen_TCgt10IFL_1deg.nc
    mk_Hansen_STD_TCgt10IFL_025deg.nc  mk_Hansen_STD_TCgt10IFL_05deg.nc
    mk_Hansen_STD_TCgt10IFL_1deg.nc
    mk_Hansen_CV_TCgt10IFL_025deg.nc   mk_Hansen_CV_TCgt10IFL_05deg.nc
    mk_Hansen_CV_TCgt10IFL_1deg.nc
    hansen_gt10_IFL_05deg_mean_std_cv.nc  — combined 0.5° time series
    (all saved to DATA_DIR)
"""

import os
import numpy as np
import skimage.measure
import pymannkendall as mk
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = r""
TC_THRESHOLD = 10

# ----- load -----
hansen = np.load(os.path.join(DATA_DIR, "target_ifl.npy"))
hansen = np.where(hansen < TC_THRESHOLD, np.nan, hansen)
print(f"Hansen shape: {hansen.shape}")

years = np.arange(2000, 2024)


def mk_test_safe(x):
    """
    Apply Mann-Kendall test pixel-wise; return [trend, slope, p-value, tau].
    Returns NaN array if insufficient data or zero variance.
    """
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


def run_mk_and_save(data, lat_arr, lon_arr, out_path, description=""):
    """Apply MK test to all pixels and save results as NetCDF."""
    n_lat, n_lon = data.shape[1], data.shape[2]
    mk_res = np.apply_along_axis(
        mk_test_safe, 0, data.reshape(data.shape[0], -1)
    )
    ds = xr.Dataset(
        {
            "trend":      (["lat", "lon"], np.flipud(mk_res[0].reshape(n_lat, n_lon))),
            "ts_slope":   (["lat", "lon"], np.flipud(mk_res[1].reshape(n_lat, n_lon))),
            "p_value":    (["lat", "lon"], np.flipud(mk_res[2].reshape(n_lat, n_lon))),
            "kendall_tau":(["lat", "lon"], np.flipud(mk_res[3].reshape(n_lat, n_lon))),
        },
        coords={"lat": lat_arr, "lon": lon_arr},
        attrs={
            "description": description,
            "units": ("trend: 1=increasing/-1=decreasing/0=no trend; "
                      "ts_slope: %TC/year; p_value: significance; "
                      "kendall_tau: correlation coefficient"),
        },
    )
    ds.to_netcdf(out_path)
    print(f"Saved {out_path}")


# ----- aggregate to coarser resolutions -----
# block_size = (years, lat_factor, lon_factor); years axis kept as-is
hansen_025 = np.stack([
    skimage.measure.block_reduce(hansen[i], (5,  5),  np.nanmean)
    for i in range(hansen.shape[0])
])
hansen_05  = np.stack([
    skimage.measure.block_reduce(hansen[i], (10, 10), np.nanmean)
    for i in range(hansen.shape[0])
])
hansen_1   = np.stack([
    skimage.measure.block_reduce(hansen[i], (20, 20), np.nanmean)
    for i in range(hansen.shape[0])
])

print(f"0.25° shape: {hansen_025.shape}")
print(f"0.5°  shape: {hansen_05.shape}")
print(f"1°    shape: {hansen_1.shape}")

# ----- compute and save TC mean trends -----
resolutions = [
    ("025deg", hansen_025),
    ("05deg",  hansen_05),
    ("1deg",   hansen_1),
]

for suffix, data in resolutions:
    lat_agg = np.linspace(-90, 90, data.shape[1])
    lon_agg = np.linspace(-180, 180, data.shape[2])
    run_mk_and_save(
        data, lat_agg, lon_agg,
        os.path.join(DATA_DIR, f"mk_Hansen_TCgt10IFL_{suffix}.nc"),
        description=f"MK trends for Hansen TC mean at {suffix} resolution",
    )

# ----- aggregate standard deviation -----
hansen_025_std = np.stack([
    skimage.measure.block_reduce(hansen[i], (5,  5),  np.nanstd)
    for i in range(hansen.shape[0])
])
hansen_05_std  = np.stack([
    skimage.measure.block_reduce(hansen[i], (10, 10), np.nanstd)
    for i in range(hansen.shape[0])
])
hansen_1_std   = np.stack([
    skimage.measure.block_reduce(hansen[i], (20, 20), np.nanstd)
    for i in range(hansen.shape[0])
])

for suffix, data in [("025deg", hansen_025_std),
                     ("05deg",  hansen_05_std),
                     ("1deg",   hansen_1_std)]:
    lat_agg = np.linspace(-90, 90, data.shape[1])
    lon_agg = np.linspace(-180, 180, data.shape[2])
    run_mk_and_save(
        data, lat_agg, lon_agg,
        os.path.join(DATA_DIR, f"mk_Hansen_STD_TCgt10IFL_{suffix}.nc"),
        description=f"MK trends for Hansen TC STD at {suffix} resolution",
    )

# ----- coefficient of variation -----
# CV = STD / mean; computed at pixel level then averaged over the block
def block_cv(year_slice, block_size):
    return skimage.measure.block_reduce(
        year_slice,
        block_size,
        lambda x, axis=None: (np.nanstd(x) / np.nanmean(x))
        if np.nanmean(x) != 0 else np.nan,
    )

# use STD / mean at coarsened scale as approximation
for suffix, mean_data, std_data in [
    ("025deg", hansen_025, hansen_025_std),
    ("05deg",  hansen_05,  hansen_05_std),
    ("1deg",   hansen_1,   hansen_1_std),
]:
    cv_data = std_data / mean_data
    lat_agg = np.linspace(-90, 90, cv_data.shape[1])
    lon_agg = np.linspace(-180, 180, cv_data.shape[2])
    run_mk_and_save(
        cv_data, lat_agg, lon_agg,
        os.path.join(DATA_DIR, f"mk_Hansen_CV_TCgt10IFL_{suffix}.nc"),
        description=f"MK trends for Hansen TC CV at {suffix} resolution",
    )

# ----- comparison plot: global mean TC across resolutions -----
data_dict = {
    "0.05°": hansen,
    "0.25°": hansen_025,
    "0.5°":  hansen_05,
    "1°":    hansen_1,
}

fig, ax = plt.subplots(figsize=(10, 5))
for label, data in data_dict.items():
    avg = np.nanmean(data.reshape(data.shape[0], -1), axis=1)
    sns.lineplot(x=years, y=avg, label=label, ax=ax)

ax.set_title("Global Mean TC at Multiple Resolutions", fontsize=12, fontweight="bold")
ax.set_xlabel("Year")
ax.set_ylabel("Mean TC (%)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ----- save combined 0.5° TC / STD / CV time series -----
# This combined file is required by 08_trend_statistics.py
cv_05 = hansen_05_std / hansen_05
lat_05 = np.linspace(-90, 90, hansen_05.shape[1])
lon_05 = np.linspace(-180, 180, hansen_05.shape[2])

ds_combined = xr.Dataset(
    {
        "TC":  (["time", "lat", "lon"], hansen_05),
        "STD": (["time", "lat", "lon"], hansen_05_std),
        "CV":  (["time", "lat", "lon"], cv_05),
    },
    coords={"time": years, "lat": lat_05, "lon": lon_05},
    attrs={"description": "Hansen TC mean, STD, and CV aggregated to 0.5° resolution"},
)
ds_combined.to_netcdf(os.path.join(DATA_DIR, "hansen_gt10_IFL_05deg_mean_std_cv.nc"))
print("Saved hansen_gt10_IFL_05deg_mean_std_cv.nc")
