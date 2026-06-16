"""
17_residuals_computation.py

Compute climate-TC disequilibrium residuals as:
    residuals = observed TC − clipped predictions

Predictions are first clipped to [0, 100] % to remove physically
implausible values.  Residuals within the model's uncertainty band
(±(RMSE + STD)) are set to zero — those pixels are considered to be
within the range the model cannot reliably resolve.

Uncertainty thresholds (RMSE + STD in % TC) from cross-validation:
    global:    8.46 + 0.02 = 8.48
    arid:      4.05 + 0.09 = 4.14
    boreal:    9.52 + 0.02 = 9.54
    temperate: 3.31 + 0.05 = 3.36
    tropics:   3.49 + 0.01 = 3.50

Inputs:
    target_ifl.npy               — observed Hansen TC (years, lat, lon)
    predictions_{area}.npy       — model predictions (lat, lon, years)
Outputs:
    masked_residuals_{area}.npy  — residuals (lat, lon, years), shape same
                                   as predictions; within-uncertainty = 0
"""

import os
import numpy as np

DATA_DIR = r""

observed = np.load(os.path.join(DATA_DIR, "target_ifl.npy"))
# reorder to (lat, lon, years) to match predictions
observed = np.transpose(observed, (1, 2, 0))
print(f"Observed shape: {observed.shape}")

# uncertainty thresholds per area (RMSE + STD from cross-validation)
areas_dict = {
    "global":    8.4622 + 0.0215,
    "arid":      4.0533 + 0.0909,
    "boreal":    9.5239 + 0.0150,
    "temperate": 3.3055 + 0.0482,
    "tropics":   3.4910 + 0.0121,
}

for area, threshold in areas_dict.items():
    preds = np.load(os.path.join(DATA_DIR, area, f"predictions_{area}.npy"))

    # clip to physical range before computing residuals
    preds_clipped = np.clip(preds, 0, 100)
    residuals     = observed - preds_clipped

    # mask residuals within the model's uncertainty band (set to 0)
    within_uncertainty = np.abs(residuals) < threshold
    masked_residuals   = np.where(within_uncertainty, 0, residuals)

    out_path = os.path.join(DATA_DIR, area, f"masked_residuals_{area}.npy")
    np.save(out_path, masked_residuals)
    print(f"{area}: saved {out_path}  "
          f"non-zero residuals: {np.sum(masked_residuals != 0):,}")

print("Done.")
