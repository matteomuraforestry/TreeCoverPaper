# Tree Cover – Climate Disequilibrium: Analysis Pipeline

This repository contains the full analysis pipeline for quantifying climate–tree cover (TC) disequilibrium using Hansen tree cover data (2000–2023) and TerraClimate climate variables, together with the scripts that reproduce the four main publication figures.

---

## Repository structure

```
├── scripts/                        # Processing pipeline — run in order 01 → 19
│   ├── 01_data_extraction.py
│   ├── 02_data_preparation.py
│   ├── 03_tc_distributions.py
│   ├── 04_biome_downsampling.py
│   ├── 05_tc_dynamics.py
│   ├── 06_semivariogram.py
│   ├── 07_resolution_sensitivity.py
│   ├── 08_trend_statistics.py
│   ├── 09_trend_maps.py
│   ├── 10_climate_filtering.py
│   ├── 11_modelling_prep.py
│   ├── 12_model_fitting.py
│   ├── 13_predictions.py
│   ├── 14_model_diagnostics.py
│   ├── 15_predictions_validation.py
│   ├── 16_residuals_computation.py
│   ├── 17_residuals_trends.py
│   ├── 18_residuals_visualization.py
│   └── 19_disequilibrium.py
├── 01_Figure1.py                   # Figure 1
├── 02_Figure2.py                   # Figure 2
├── 03_Figure3.py                   # Figure 3
└── 04_Figure4.py                   # Figure 4
```

---

## Getting started

1. **Set paths.** Every script exposes `DATA_DIR` (and `OUTPUT_DIR` / `OUTPUT_PATH`) near the top. Set these to your local data and output folders before running.
2. **Run in order.** The pipeline scripts are numbered 01–19; each step reads files produced by earlier steps.
3. **Google Earth Engine (script 01 only).** Fill in `PROJECT_ID` and authenticate with GEE only if you want to re-download the raw data. All other scripts work offline with the provided `.npy` and `.nc` files.

---

## Pipeline scripts (`scripts/`)

| Script | Description | Key inputs | Key outputs |
|--------|-------------|------------|-------------|
| `01_data_extraction.py` | Downloads Hansen TC and TerraClimate climate data (2000–2023) from Google Earth Engine at 0.05° resolution; stacks 14 bands (TC, 6 climate means, 6 climate σ, year index) | GEE assets | `data_YYYY.npy` per year |
| `02_data_preparation.py` | Applies 10 % TC threshold to the IFL-masked Hansen array; pixels below threshold are set to NaN | `target_ifl.npy` | `target_ifl.npy` (filtered, overwrite) |
| `03_tc_distributions.py` | Histograms and statistics of the arcsine-sqrt-transformed TC distribution globally and per biome | `target_ifl.npy`, `biomes_kg_005.npy` | Screen plots |
| `04_biome_downsampling.py` | Mode-aggregates the Köppen-Geiger biome map from 0.05° to 0.5° and 1°; polar biome (code 5) excluded | `biomes_kg_005.npy` | `biomes_kg_05.npy`, `biomes_kg_1.npy` |
| `05_tc_dynamics.py` | Area-weighted TC transition matrices (2000 → 2023) between 10 %-point bins; latitude-corrected pixel areas | `target_ifl.npy`, `biomes_kg_005.npy` | Transition plots |
| `06_semivariogram.py` | Empirical semivariogram analysis; the 95 %-sill range gives the per-biome spatial autocorrelation length used later for confidence interval corrections | `target_ifl.npy`, `biomes_kg_005.npy` | Autocorrelation length estimates and semivariogram plots |
| `07_resolution_sensitivity.py` | Mann-Kendall / Theil-Sen trends of TC mean, STD, and CV at 0.05°, 0.25°, 0.5°, 1°; saves the 0.5° time series used downstream | `target_ifl.npy` | `mk_Hansen_*.nc` at each resolution; `hansen_gt10_IFL_05deg_mean_std_cv.nc` |
| `08_trend_statistics.py` | Biome-level TC trends with spatially corrected 95 % CI, using biome-specific effective sample sizes derived from autocorrelation lengths | `hansen_gt10_IFL_05deg_mean_std_cv.nc`, `biomes_kg_05.npy` | Printed MK results and time-series CI plots |
| `09_trend_maps.py` | Robinson-projection maps of Theil-Sen slopes and Kendall's τ for TC mean, STD, and CV | `mk_Hansen_*.nc` | Maps on screen |
| `10_climate_filtering.py` | Applies IFL mask and TC > 10 % threshold to the annual 14-band climate stacks | `data_YYYY.npy`, `ifl2020_005.npy` | `data_YYYY_ifl_gt10.npy` per year |
| `11_modelling_prep.py` | Checkerboard non-adjacent sampling (70 % fraction) to build training sets for global and four biome-specific domains | `data_YYYY_ifl_gt10.npy`, `biomes_kg_005.npy` | `X_{area}.npy`, `y_{area}.npy` per area |
| `12_model_fitting.py` | Fits `HistGradientBoostingRegressor` per area using Bayesian hyperparameter search (30 iterations, 3 × 5 RepeatedKFold); computes permutation importance, SHAP values, and ICE curves | `X_{area}.npy`, `y_{area}.npy` | `best_model_{area}.joblib`, SHAP and importance arrays |
| `13_predictions.py` | Parallelised (8 workers) spatially complete predictions for all years and areas; biome models predict within their domain only; global model covers all land | `data_YYYY.npy`, `best_model_{area}.joblib`, `biomes_kg_005.npy` | `predictions_{area}.npy` per area |
| `14_model_diagnostics.py` | PDF diagnostic report: permutation importance bar chart, SHAP violin/bar/clustered plots, and ICE/PDP curves for all 12 climate features | Model output arrays per area | Diagnostic PDFs at 300 DPI |
| `15_predictions_validation.py` | Reports the fraction of out-of-range predictions (< 0 % or > 100 %) per year and area to validate clipping decisions before computing residuals | `predictions_{area}.npy` | Printed sanity-check statistics |
| `16_residuals_computation.py` | Residuals = observed − clipped(predicted); pixels within the model uncertainty band (biome-specific RMSE + STD threshold) are zeroed out | `target_ifl.npy`, `predictions_{area}.npy` | `masked_residuals_{area}.npy` per area |
| `17_residuals_trends.py` | Mann-Kendall / Theil-Sen trends of residuals at 0.05° (pixel level) and at 0.5° (block-aggregated mean and STD) | `masked_residuals_{area}.npy` | `mk_Residuals_{area}_005deg.nc`, `mk_Residuals_MeanSTD_{area}_05deg.nc` |
| `18_residuals_visualization.py` | Climatological residual maps (5 × 5 smoothed), time-series with autocorrelation-corrected 95 % CI, 4-panel publication figure, and distribution histograms | `masked_residuals_global.npy`, `mk_Residuals_MeanSTD_global_05deg.nc`, biome maps | Screen figures |
| `19_disequilibrium.py` | Reports global mean climatological disequilibrium (d_clim, 2000–2023) and the 2023 snapshot (d_2023), both with spatially corrected 95 % CI (110-pixel autocorrelation length) | `predictions_global.npy`, `target_ifl.npy` | Printed estimates with confidence intervals |

---

## Figure scripts

Each figure script is self-contained and reproduces one publication figure. Set `DATA_DIR` and `OUTPUT_PATH` at the top of each file before running.

| Script | Figure | Description |
|--------|--------|-------------|
| `01_Figure1.py` | Figure 1 | Heatmaps of annual TC distribution shifts relative to the year-2000 baseline, globally and per biome. Rows = years, columns = 10 %-wide TC bins; colour shows the deviation from the 2000 distribution. |
| `02_Figure2.py` | Figure 2 | Area-weighted TC transition matrices (2000 → 2023) showing how forest pixels moved between 10 %-wide TC bins. Background colour encodes total forest area (log scale); annotated numbers are row-conditional percentages. |
| `03_Figure3.py` | Figure 3 | Spatial trends in TC mean and variability: global Theil-Sen slope maps at 0.5° (panels a–b), biome bar charts with 95 % CI (panel c), and a hexbin scatter of δTC_std vs. δTC_avg with Pearson r (panel d). |
| `04_Figure4.py` | Figure 4 | Spatial trends in TC residuals: global Theil-Sen slope maps at 0.5° (panels a–b), biome bar charts with 95 % CI (panel c), and a latitude-weighted forest-area pie chart by biome (panel d). |

---

## Data description

| File | Description |
|------|-------------|
| `target_ifl.npy` | Hansen-derived annual tree cover (%) at 0.05°, IFL-masked; shape (24, 3600, 7200), years 2000–2023 |
| `data_YYYY.npy` | 14-band annual stacks: TC + 6 TerraClimate means + 6 TerraClimate σ + year index |
| `data_YYYY_ifl_gt10.npy` | Same as above, filtered to IFL pixels with TC > 10 % |
| `ifl2020_005.npy` | Intact Forest Landscape (IFL) mask at 0.05° |
| `biomes_kg_005.npy` | Köppen-Geiger biome classification at 0.05° (1 = Tropics, 2 = Arid, 3 = Temperate, 4 = Boreal, 5 = Polar) |
| `biomes_kg_05.npy` | Same, mode-aggregated to 0.5° |
| `hansen_gt10_IFL_05deg_mean_std_cv.nc` | 0.5° time series of annual TC mean, STD, and CV (produced by script 07) |
| `mk_Hansen_*.nc` | Pre-computed Mann-Kendall / Theil-Sen results for TC mean, STD, and CV at 0.5° |
| `best_model_{area}.joblib` | Serialised fitted models per area (global, tropics, arid, temperate, boreal) |
| `predictions_{area}.npy` | Spatially complete annual TC predictions per area |
| `masked_residuals_{area}.npy` | Annual TC residuals (observed − predicted), uncertainty-masked, per area |
| `mk_Residuals_*_{area}_*.nc` | Pre-computed MK / Theil-Sen trends of residuals per area and resolution |

---

## Dependencies

```
numpy  xarray  scipy  matplotlib  seaborn  cartopy  scikit-image
scikit-learn  scikit-optimize  shap  joblib  pymannkendall  tqdm
earthengine-api  geemap          # required only for script 01
```

Install via pip:

```bash
pip install numpy xarray scipy matplotlib seaborn cartopy scikit-image \
    scikit-learn scikit-optimize shap joblib pymannkendall tqdm \
    earthengine-api geemap
```
