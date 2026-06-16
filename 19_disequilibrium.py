"""
20_disequilibrium.py

Quantify the climate-tree cover disequilibrium as the difference between
observed and model-predicted tree cover.

Two estimates are reported:
  d_clim  — mean disequilibrium across the full 2000-2023 climatology
  d_2023  — disequilibrium in the most recent year (2023)

95% confidence intervals account for spatial autocorrelation:
    effective n = n_valid_pixels / autocorr_length²

where autocorr_length = 110 pixels (~550 km at 0.05°) based on the
semivariogram analysis (script 06).

Inputs:
    predictions_global.npy  — model predictions (lat, lon, years)
    target_ifl.npy          — observed Hansen TC (years, lat, lon)
Outputs:
    Printed estimates with 95% CI.
"""

import numpy as np
from scipy import stats

DATA_DIR       = r""
AUTOCORR_PX    = 110   # autocorrelation length in pixels (from semivariogram)
LAST_YEAR_IDX  = 23    # index for 2023 (0-based from 2000)

# ----- load -----
predictions  = np.load(os.path.join(DATA_DIR, "global", "predictions_global.npy"))
observations = np.load(os.path.join(DATA_DIR, "target_ifl.npy")).transpose(1, 2, 0)

print(f"Predictions:  {predictions.shape}")
print(f"Observations: {observations.shape}")

# ----- align: keep only pixels valid across all years in both arrays -----
valid_mask = (
    ~np.isnan(predictions).any(axis=2)
    & ~np.isnan(observations).any(axis=2)
)
preds_aligned = np.where(valid_mask[:, :, np.newaxis], predictions,  np.nan)
obs_aligned   = np.where(valid_mask[:, :, np.newaxis], observations, np.nan)

n_valid = int(np.sum(valid_mask))
n_eff   = n_valid / (AUTOCORR_PX ** 2)
print(f"\nValid pixels: {n_valid:,}  |  Effective n: {n_eff:.1f}")

# ----- climatological disequilibrium (2000-2023) -----
obs_mean  = np.nanmean(obs_aligned,  axis=2)
pred_mean = np.nanmean(preds_aligned, axis=2)
diff_clim = obs_mean - pred_mean

d_clim = np.nanmean(diff_clim)
se_clim = np.nanstd(diff_clim) / np.sqrt(n_eff)
ci_clim = stats.t.interval(
    0.95, df=n_eff - 1, loc=d_clim, scale=se_clim
)

print(f"\nClimatological disequilibrium (2000-2023):")
print(f"  d_clim = {d_clim:.3f} %TC")
print(f"  95% CI  [{ci_clim[0]:.3f}, {ci_clim[1]:.3f}]")

# ----- 2023 disequilibrium -----
diff_2023 = obs_aligned[:, :, LAST_YEAR_IDX] - preds_aligned[:, :, LAST_YEAR_IDX]
d_2023    = np.nanmean(diff_2023)
n_2023    = np.sum(~np.isnan(diff_2023))
n_eff_23  = n_2023 / (AUTOCORR_PX ** 2)
se_2023   = np.nanstd(diff_2023) / np.sqrt(n_eff_23)
ci_2023   = stats.t.interval(
    0.95, df=n_eff_23 - 1, loc=d_2023, scale=se_2023
)

print(f"\n2023 disequilibrium:")
print(f"  d_2023 = {d_2023:.3f} %TC")
print(f"  95% CI  [{ci_2023[0]:.3f}, {ci_2023[1]:.3f}]")
