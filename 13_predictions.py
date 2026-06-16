"""
14_predictions.py

Generate spatially complete predictions for all grid cells using the fitted
models.  Biome models predict only within their respective domain; the global
model covers all land pixels.

Processing is parallelised across years (8 workers by default) to reduce
runtime.  Adjust N_JOBS to match available CPU cores.

Inputs:
    data_YYYY.npy               — raw annual arrays (TC + predictors)
    best_model_{area}.joblib    — fitted models per area
    biomes_kg_005.npy           — biome mask at 0.05°
Outputs:
    predictions_{area}.npy  — array of shape (lat, lon, years)
    saved to DATA_DIR/{area}/
"""

import os
import gc
import time
import numpy as np
import joblib
from joblib import Parallel, delayed

DATA_DIR = r""
N_JOBS   = 8   # parallel workers — adjust to available cores

biomes     = np.load(os.path.join(DATA_DIR, "biomes_kg_005.npy"))
data_files = sorted(f for f in os.listdir(DATA_DIR) if f.startswith("data_2"))

areas_dict = {"tropics": 1, "arid": 2, "temperate": 3, "boreal": 4}

print(f"Years to process: {len(data_files)}")
print(f"Parallel workers: {N_JOBS}")


# ----- prediction functions -----

def predict_biome_year(data_file, main_dir, biomes_arr, biome_val, model):
    """Predict TC for one year, masked to a single biome."""
    predictors = np.load(os.path.join(main_dir, data_file))[1:]
    n_lat, n_lon = predictors.shape[1], predictors.shape[2]

    preds_masked = np.where(biomes_arr == biome_val, predictors, np.nan)
    preds_flat   = preds_masked.transpose(1, 2, 0).reshape(-1, preds_masked.shape[0])

    return model.predict(preds_flat), n_lat, n_lon


def predict_global_year(data_file, main_dir, model):
    """Predict TC for one year across all land pixels."""
    predictors = np.load(os.path.join(main_dir, data_file))[1:]
    n_lat, n_lon = predictors.shape[1], predictors.shape[2]

    preds_flat = predictors.transpose(1, 2, 0).reshape(-1, predictors.shape[0])
    return model.predict(preds_flat), n_lat, n_lon


# ----- biome models -----
t_start = time.time()

for biome_name, biome_val in areas_dict.items():
    area_dir    = os.path.join(DATA_DIR, biome_name)
    best_model  = joblib.load(os.path.join(area_dir, f"best_model_{biome_name}.joblib"))

    with joblib.parallel_backend("loky", n_jobs=N_JOBS):
        results = Parallel(verbose=5)(
            delayed(predict_biome_year)(f, DATA_DIR, biomes, biome_val, best_model)
            for f in data_files
        )

    preds_list       = [r[0] for r in results]
    n_lat, n_lon     = results[0][1], results[0][2]
    preds_array      = (np.stack(preds_list, axis=1)
                         .reshape(n_lat, n_lon, len(data_files)))

    out_path = os.path.join(area_dir, f"predictions_{biome_name}.npy")
    np.save(out_path, preds_array)
    print(f"{biome_name}: saved {out_path}  shape={preds_array.shape}")

    del best_model, results, preds_list, preds_array
    gc.collect()

# ----- global model -----
area_dir   = os.path.join(DATA_DIR, "global")
best_model = joblib.load(os.path.join(area_dir, "best_model_global.joblib"))

with joblib.parallel_backend("loky", n_jobs=N_JOBS):
    results = Parallel(verbose=5)(
        delayed(predict_global_year)(f, DATA_DIR, best_model)
        for f in data_files
    )

preds_list   = [r[0] for r in results]
n_lat, n_lon = results[0][1], results[0][2]
preds_array  = (np.stack(preds_list, axis=1)
                 .reshape(n_lat, n_lon, len(data_files)))

out_path = os.path.join(area_dir, "predictions_global.npy")
np.save(out_path, preds_array)
print(f"global: saved {out_path}  shape={preds_array.shape}")

elapsed = time.time() - t_start
print(f"\nTotal time: {elapsed / 60:.1f} min")
