"""
02_data_preparation.py

Prepare the Hansen tree cover dataset for analysis.
Loads target_ifl.npy (Hansen TC, already IFL-masked), applies a minimum
TC threshold of 10%, and saves the filtered array.

Input:  target_ifl.npy  — Hansen TC [years x lat x lon], IFL-masked
Output: target_ifl.npy  — same file, overwritten with TC < 10% set to NaN
"""

import os
import numpy as np

DATA_DIR = r""
TC_THRESHOLD = 10  # minimum tree cover to retain (%)

hansen = np.load(os.path.join(DATA_DIR, "target_ifl.npy"))
print(f"Loaded Hansen TC: shape={hansen.shape}")

# round to integer values (consistent with original 1% resolution)
hansen = np.round(hansen).astype(float)

# mask pixels below the TC threshold
hansen = np.where(hansen < TC_THRESHOLD, np.nan, hansen)

print(f"Valid pixels after TC ≥ {TC_THRESHOLD}% filter: {np.sum(~np.isnan(hansen)):,}")
print(f"TC range: {np.nanmin(hansen):.1f} – {np.nanmax(hansen):.1f} %")

np.save(os.path.join(DATA_DIR, "target_ifl.npy"), hansen)
print("Saved target_ifl.npy")
