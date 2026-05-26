# Reproducible Code — Tree Cover Climate Disequilibrium

This repository contains the Python scripts used to produce the four main figures
of the manuscript on global tree-cover dynamics and climate disequilibrium (2000–2023).
Each script is self-contained and produces a single publication-quality figure as a PDF.

---

## Figures

| Script | Figure | Description |
|--------|--------|-------------|
| `01_Figure1.py` | Figure 1 | Heatmaps of shifts in the tree-cover distribution relative to the year-2000 baseline, globally and by biome |
| `02_Figure2.py` | Figure 2 | Area-weighted TC transition matrices (2000→2023) and a summary bar chart of change magnitude |
| `03_Figure3.py` | Figure 3 | Spatial trends in mean and variability of TC, with biome-level bar charts and a pixel-level scatter |
| `04_Figure4.py` | Figure 4 | Spatial trends in TC residuals (observed minus climate-predicted), with biome bar charts and a forest-area pie chart |

---

## Data

Set the `DATA_DIR` variable at the top of each script to the folder containing the files below.

| File | Used by | Description |
|------|---------|-------------|
| `target_ifl.npy` | Figs 1–4 | Float32 array `(24, H, W)` of annual Hansen-derived TC values at 0.05° resolution, 2000–2023 |
| `biomes_kg_005.npy` | Figs 1–2 | Köppen-Geiger biome class per pixel at 0.05° (`1` Tropics · `2` Arid · `3` Temperate · `4` Boreal) |
| `biomes_kg_05.npy` | Figs 3–4 | Same classification aggregated to 0.5° |
| `mk_Hansen_TCgt10IFL_05deg.nc` | Fig 3 | Pre-computed Mann-Kendall / Theil-Sen results for mean TC at 0.5° |
| `mk_Hansen_STD_TCgt10IFL_05deg.nc` | Fig 3 | Same for TC standard deviation |
| `global/mk_Residuals_MeanSTD_global_05deg.nc` | Fig 4 | Pre-computed Mann-Kendall / Theil-Sen results for residual mean and std at 0.5° |
| `global/masked_residuals_global.npy` | Fig 4 | Float32 array `(H, W, 24)` of annual TC residuals (observed − climate-predicted) at 0.05° |

> The Mann-Kendall trend analysis and the climate-model residuals are computed in
> separate upstream notebooks and are not reproduced here.

---

## Dependencies

```bash
pip install numpy pandas matplotlib seaborn scipy xarray cartopy scikit-image
