"""
04_biome_downsampling.py

Downsample the Koppen-Geiger biome map from 0.05° to 0.5° and 1° using
mode aggregation (most frequent non-NaN class within each block).

Polar regions (biome code 5) are excluded before downsampling.

Inputs:
    biomes_kg_005.npy   — biome map at 0.05° resolution
Outputs:
    biomes_kg_05.npy    — biome map at 0.5°
    biomes_kg_1.npy     — biome map at 1°
"""

import os
import numpy as np
import skimage.measure
from scipy import stats

DATA_DIR = r""

biomes_005 = np.load(os.path.join(DATA_DIR, "biomes_kg_005.npy"))

# exclude polar biome (code 5)
biomes_005 = np.where(biomes_005 == 5, np.nan, biomes_005)

print(f"Input shape (0.05°): {biomes_005.shape}")
print(f"Unique biome codes: {np.unique(biomes_005[~np.isnan(biomes_005)])}")


def mode_ignoring_nan(block, axis=None):
    """Return most frequent non-NaN value in a block; NaN if all values are NaN."""
    if axis is not None:
        new_shape = block.shape[:len(block.shape) - len(axis)]
        block_reshaped = block.reshape(new_shape + (-1,))
        result = np.empty(new_shape)
        for idx in np.ndindex(new_shape):
            values = block_reshaped[idx]
            valid = values[~np.isnan(values)]
            if len(valid) == 0:
                result[idx] = np.nan
            else:
                result[idx] = stats.mode(valid, keepdims=False).mode
        return result
    else:
        flat = block.flatten()
        valid = flat[~np.isnan(flat)]
        if len(valid) == 0:
            return np.nan
        return stats.mode(valid, keepdims=False).mode


# 0.05° → 0.5°  (10 × 10 blocks)
biomes_05 = skimage.measure.block_reduce(
    biomes_005, block_size=(10, 10), func=mode_ignoring_nan, cval=np.nan
)

# 0.05° → 1°  (20 × 20 blocks)
biomes_1 = skimage.measure.block_reduce(
    biomes_005, block_size=(20, 20), func=mode_ignoring_nan, cval=np.nan
)

print(f"Downsampled shape (0.5°): {biomes_05.shape}")
print(f"Downsampled shape (1°):   {biomes_1.shape}")

np.save(os.path.join(DATA_DIR, "biomes_kg_05.npy"), biomes_05)
np.save(os.path.join(DATA_DIR, "biomes_kg_1.npy"),  biomes_1)
print("Saved biomes_kg_05.npy and biomes_kg_1.npy")
