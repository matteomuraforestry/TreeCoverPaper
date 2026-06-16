"""
11_modelling_prep.py

Prepare training datasets for global and biome-specific models using a
checkerboard non-adjacent pixel sampling strategy to reduce spatial
autocorrelation bias.  70% of valid pixels per year are drawn at random.

Inputs:
    data_YYYY_ifl_gt10.npy  — filtered annual arrays (14 bands)
    biomes_kg_005.npy       — biome map at 0.05°
Outputs (one pair per area, saved to DATA_DIR/{area}/):
    y_{area}.npy  — TC targets
    X_{area}.npy  — feature matrix
    where area ∈ {global, tropics, arid, temperate, boreal}
"""

import os
import numpy as np

DATA_DIR = r""
FRACTION   = 0.70   # fraction of valid pixels to sample per year
SEED       = 86

np.random.seed(SEED)

biomes = np.load(os.path.join(DATA_DIR, "biomes_kg_005.npy"))

biomes_dict = {1: "tropics", 2: "arid", 3: "temperate", 4: "boreal"}

# checkerboard mask (same as global prep)
mask = np.zeros((3600, 7200), dtype=bool)
mask[1::2, ::2] = True

data_files = sorted(f for f in os.listdir(DATA_DIR)
                    if f.startswith("data_") and f.endswith("_ifl_gt10.npy"))
print(f"Found {len(data_files)} annual files")


def extract_samples(fname, biome_idx=None):
    """
    Apply checkerboard mask and optional biome filter, then sample
    FRACTION of valid pixels.  Returns (y, X).
    """
    data = np.load(os.path.join(DATA_DIR, fname))
    filtered = np.where(mask, data, np.nan)

    if biome_idx is not None:
        filtered = np.where(biomes == biome_idx, filtered, np.nan)

    valid_mask = ~np.isnan(filtered).any(axis=0)
    valid_rows, valid_cols = np.where(valid_mask)
    n_valid = len(valid_rows)

    n_samples = max(1, round(n_valid * FRACTION))
    chosen = np.random.choice(n_valid, size=n_samples, replace=False)

    sampled = filtered[:, valid_rows[chosen], valid_cols[chosen]]
    return sampled[0], sampled[1:]


def build_and_save(biome_code, biome_name):
    results = [extract_samples(f, biome_idx=biome_code) for f in data_files]
    y = np.concatenate([r[0] for r in results])
    X = np.concatenate([r[1] for r in results], axis=1).T
    print(f"{biome_name}: y={y.shape}  X={X.shape}")
    # save inside the per-area subdirectory expected by 12_model_fitting.py
    area_dir = os.path.join(DATA_DIR, biome_name)
    os.makedirs(area_dir, exist_ok=True)
    np.save(os.path.join(area_dir, f"y_{biome_name}.npy"), y)
    np.save(os.path.join(area_dir, f"X_{biome_name}.npy"), X)
    print(f"  Saved to {area_dir}/")


# global (no biome filter)
print("Processing global …")
build_and_save(None, "global")

# per-biome
for code, name in biomes_dict.items():
    print(f"Processing {name} …")
    build_and_save(code, name)
