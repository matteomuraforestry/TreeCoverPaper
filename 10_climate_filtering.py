"""
10_climate_filtering.py

Apply the IFL mask and a TC > 10% threshold to the annual data arrays
(data_YYYY.npy) that combine tree cover and climate predictors.

Each input file has shape (14, 3600, 7200):
    band 0   — Hansen tree cover (%)
    bands 1-13 — climate predictors

Pixels where TC ≤ 10% or outside the IFL mask are set to NaN.
Output files are named data_YYYY_ifl_gt10.npy.

Inputs:
    data_YYYY.npy    — annual stacked arrays (14 bands)
    ifl2020_005.npy  — intact forest landscape mask at 0.05°
Outputs:
    data_YYYY_ifl_gt10.npy  — filtered arrays (same shape, non-forest = NaN)
"""

import os
import numpy as np

DATA_DIR = r""
TC_BAND  = 0     # index of the TC band in the stacked array
TC_THRESHOLD = 10

# ----- load IFL mask -----
ifl = np.load(os.path.join(DATA_DIR, "ifl2020_005.npy"))
print(f"IFL mask shape: {ifl.shape}")

# ----- filter each annual file -----
data_files = sorted(f for f in os.listdir(DATA_DIR) if f.startswith("data_2"))

for fname in data_files:
    out_name = fname.replace(".npy", "_ifl_gt10.npy")
    out_path = os.path.join(DATA_DIR, out_name)
    print(f"Processing {fname} → {out_name}")

    data = np.load(os.path.join(DATA_DIR, fname))

    # mask where TC ≤ threshold or pixel is outside IFL
    valid = (data[TC_BAND] > TC_THRESHOLD) & (ifl > 0)
    filtered = np.where(valid, data, np.nan)

    np.save(out_path, filtered)

print("Done.")
