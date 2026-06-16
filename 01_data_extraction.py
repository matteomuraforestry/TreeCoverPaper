"""
01_data_extraction.py

Extract Hansen tree cover and TerraClimate climate data from Google Earth Engine
for the period 2000-2023 at 0.05° resolution.

Each year produces a 14-band array (TC + 6 climate means + 6 climate std devs + year)
saved as data_YYYY.npy.

Requirements: earthengine-api, geemap, xarray, numpy
Run: ee.Authenticate() once before first use.
"""

import os
import gc
import time

import ee
import geemap
import numpy as np

# ----- configuration -----
OUTPUT_DIR = r""
PROJECT_ID = ""  # your GEE project ID
EPSG = "EPSG:4326"
TC_THRESHOLD = 0      # keep pixels with TC > 0% during extraction
RES_DEG = 0.05        # output spatial resolution (degrees)
CLIMATIC_WINDOW = 30  # rolling window for climate averaging (years)
YEARS = list(range(2000, 2024))

# ----- helper functions -----

def reproject_and_reduce(ee_image):
    """Reproject to target CRS and coarsen to climate grid resolution."""
    reprojected = ee_image.reproject(crs=projection)
    reduced = reprojected.reduceResolution(reducer=reducer, maxPixels=1024)
    return reduced


def mask_collection(ee_image):
    """Mask climate pixels where tree cover is below threshold."""
    return ee_image.updateMask(mask_tc)


def compute_tavg(img):
    """Add mean temperature band from tmin and tmax."""
    tmmn = img.select("tmmn")
    tmmx = img.select("tmmx")
    tavg = tmmn.add(tmmx).divide(2).rename("tavg")
    return img.addBands(tavg)


def avg_monthly_precipitation(yr, start_yr, pr_collection):
    """Compute mean annual precipitation for a single year."""
    start_date = ee.Date.fromYMD(yr, 1, 1)
    end_date = ee.Date.fromYMD(yr, 12, 31)
    yearly_col = pr_collection.filterDate(start_date, end_date)
    yearly_mean = yearly_col.reduce(ee.Reducer.mean())
    yearly_precip = yearly_mean.multiply(12)
    return yearly_precip.set("year", yr)


# ----- initialise GEE -----
ee.Authenticate()
ee.Initialize(
    opt_url="https://earthengine-highvolume.googleapis.com",
    project=PROJECT_ID,
)

# ----- main extraction loop -----
for year in YEARS:
    start_year = year - (CLIMATIC_WINDOW - 1)
    print(f"\n=== Year {year} (climate window {start_year}-{year}) ===")

    # tree cover
    tree_cover_ee = (
        ee.ImageCollection("users/deepasomasundaram/TC500m")
        .select("TC")
        .filter(ee.Filter.eq("system:index", f"TC500_{year}"))
        .first()
    )
    time.sleep(10)

    # TerraClimate (30-year window)
    climate_ee = (
        ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE")
        .select(["def", "pr", "soil", "srad", "tmmn", "tmmx", "vpd"])
        .filter(ee.Filter.rangeContains("system:index", f"{start_year}01", f"{year}12"))
    )
    time.sleep(10)

    # match spatial resolution to TerraClimate grid
    spat_res = climate_ee.select("soil").first().projection().nominalScale()
    projection = ee.Projection(EPSG).atScale(spat_res)
    reducer = ee.Reducer.mean()

    # reproject and coarsen TC
    reprojected_tc = tree_cover_ee.reproject(crs=projection)
    reduced_tc = reprojected_tc.reduceResolution(reducer=reducer, maxPixels=1024)
    time.sleep(10)

    # mask TC pixels below threshold
    mask_tc = reduced_tc.gt(TC_THRESHOLD)
    tree_cover_masked = reduced_tc.updateMask(mask_tc)
    time.sleep(10)

    # TC → xarray → numpy
    tc_xr = geemap.ee_to_xarray(tree_cover_masked, crs=EPSG, scale=RES_DEG)
    time.sleep(10)
    target_2d = np.rot90(tc_xr["TC"].isel(time=0).drop_vars("time"))
    time.sleep(10)

    # --- climate averages ---
    reduced_climate = climate_ee.map(reproject_and_reduce)
    reduced_climate_masked = reduced_climate.map(mask_collection)
    climate_avg = reduced_climate_masked.mean()
    time.sleep(10)

    clim_avg_xr = geemap.ee_to_xarray(climate_avg, crs=EPSG, scale=RES_DEG)
    time.sleep(10)

    tmmn_avg = np.rot90(np.squeeze(clim_avg_xr["tmmn"].data)) * 0.1
    tmmx_avg = np.rot90(np.squeeze(clim_avg_xr["tmmx"].data)) * 0.1
    tavg_avg = np.nanmean(np.array([tmmn_avg, tmmx_avg]), axis=0)
    def_avg  = np.rot90(np.squeeze(clim_avg_xr["def"].data))  * 0.1
    pr_avg   = np.rot90(np.squeeze(clim_avg_xr["pr"].data))   * 12
    srad_avg = np.rot90(np.squeeze(clim_avg_xr["srad"].data)) * 0.1
    vpd_avg  = np.rot90(np.squeeze(clim_avg_xr["vpd"].data))  * 0.01
    soil_avg = np.rot90(np.squeeze(clim_avg_xr["soil"].data)) * 0.1
    time.sleep(10)

    # --- climate standard deviations ---
    # temperature std (computed from monthly tavg)
    climate_with_tavg = reduced_climate_masked.map(compute_tavg)
    tavg_std_xr = geemap.ee_to_xarray(
        climate_with_tavg.select("tavg").reduce(ee.Reducer.stdDev()),
        crs=EPSG, scale=0.05,
    )
    tavg_std = np.rot90(np.squeeze(tavg_std_xr["tavg_stdDev"].data)) * 0.1
    time.sleep(10)

    climate_std = reduced_climate_masked.reduce(ee.Reducer.stdDev())
    clim_std_xr = geemap.ee_to_xarray(climate_std, crs=EPSG, scale=RES_DEG)
    time.sleep(10)

    def_std  = np.rot90(np.squeeze(clim_std_xr["def_stdDev"].data))  * 0.1
    soil_std = np.rot90(np.squeeze(clim_std_xr["soil_stdDev"].data)) * 0.1
    srad_std = np.rot90(np.squeeze(clim_std_xr["srad_stdDev"].data)) * 0.1
    vpd_std  = np.rot90(np.squeeze(clim_std_xr["vpd_stdDev"].data))  * 0.01
    time.sleep(10)

    # precipitation std across years in the window
    pr_collection = climate_ee.select("pr")
    yearly_precip = ee.ImageCollection(
        [avg_monthly_precipitation(yr, start_year, pr_collection)
         for yr in range(start_year, year)]
    )
    pr_std_xr = geemap.ee_to_xarray(
        yearly_precip.reduce(ee.Reducer.stdDev()), crs=EPSG, scale=RES_DEG
    )
    pr_std = np.rot90(np.squeeze(pr_std_xr["pr_mean_stdDev"].data))
    time.sleep(10)

    # year index layer (for model to distinguish temporal context)
    year_2d = np.where(
        ~np.isnan(target_2d),
        np.full(target_2d.shape, year, dtype=float),
        np.nan,
    )

    # stack all 14 bands: TC, 6 means, 6 stds, year
    data_2d = np.stack([
        target_2d,
        def_avg, pr_avg, soil_avg, srad_avg, tavg_avg, vpd_avg,
        def_std, pr_std, soil_std, srad_std, tavg_std, vpd_std,
        year_2d,
    ])

    out_path = os.path.join(OUTPUT_DIR, f"data_{year}.npy")
    np.save(out_path, data_2d)
    print(f"Saved {out_path}  shape={data_2d.shape}")

    # free memory before next iteration
    del (tree_cover_ee, climate_ee, reduced_tc, tree_cover_masked,
         tc_xr, target_2d, reduced_climate, reduced_climate_masked,
         climate_avg, clim_avg_xr, tmmn_avg, tmmx_avg, tavg_avg,
         def_avg, pr_avg, srad_avg, vpd_avg, soil_avg,
         climate_with_tavg, tavg_std_xr, tavg_std,
         climate_std, clim_std_xr, def_std, soil_std, srad_std, vpd_std,
         pr_collection, yearly_precip, pr_std_xr, pr_std,
         year_2d, data_2d)
    gc.collect()
