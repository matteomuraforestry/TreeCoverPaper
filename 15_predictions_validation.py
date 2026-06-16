"""
16_predictions_validation.py

Sanity-check model predictions by reporting the fraction of pixels with
out-of-range values (negative or > 100%) per year and per area.

Tree cover is bounded [0, 100] % by definition.  A small percentage of
boundary violations is expected from gradient boosting (no clipping in the
objective).  This script quantifies that and informs the clipping threshold
applied before residual computation.

Inputs:
    predictions_{area}.npy  — raw predictions (lat, lon, years)
Outputs:
    Printed summary statistics per year and area.
"""

import os
import numpy as np

DATA_DIR = r""
AREAS    = ["global", "arid", "boreal", "temperate", "tropics"]
YEARS    = list(range(2000, 2024))

for area in AREAS:
    data = np.load(os.path.join(DATA_DIR, area, f"predictions_{area}.npy"))
    print(f"\n{area.upper()}  (shape: {data.shape})")
    print(f"{'Year':<6} {'Neg (%)':>10} {'>100% (%)':>12} {'Min':>8} {'Max':>8}")
    print("-" * 50)
    for i, year in enumerate(YEARS):
        yr_data = data[:, :, i]
        n_valid = np.nansum(~np.isnan(yr_data))
        n_neg   = np.nansum(yr_data < 0)
        n_over  = np.nansum(yr_data > 100)
        print(f"{year:<6} {100 * n_neg / n_valid:>9.3f}% "
              f"{100 * n_over / n_valid:>11.3f}%  "
              f"{np.nanmin(yr_data):>7.2f}  {np.nanmax(yr_data):>7.2f}")
