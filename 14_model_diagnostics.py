"""
15_model_diagnostics.py

Generate diagnostic plots for all five fitted models:
  - Permutation feature importance (bar chart with ± std)
  - SHAP violin plot (layered, sorted by mean |SHAP|)
  - SHAP bar plot (mean |SHAP| per feature)
  - SHAP bar plot with hierarchical clustering (correlation-based)
  - ICE / PDP plots for all 12 climate features

All plots are saved as PDF at 300 DPI to DATA_DIR/{area}/.

Inputs (per area in DATA_DIR/{area}/):
    best_model_{area}.joblib
    permutation_importance_{area}.csv
    shap_values_{area}.npy, shap_base_value_{area}.npy
    ice_data_{area}.joblib, X_ice_sample_{area}.npy
    X_{area}.npy, y_{area}.npy
    names.npy
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib

DATA_DIR = r""
AREAS    = ["arid", "tropics", "temperate", "boreal", "global"]
DPI      = 300

feature_names = np.load(os.path.join(DATA_DIR, "names.npy"))[1:]

# units for ICE axis labels
UNITS = {
    "CWD_avg": "(mm)",  "PR_avg": "(mm)",   "SM_avg": "(mm)",
    "SRAD_avg": "(W/m²)", "T_avg": "(°C)",  "VPD_avg": "(kPa)",
    "CWD_std": "(mm)",  "PR_std": "(mm)",   "SM_std": "(mm)",
    "SRAD_std": "(W/m²)", "T_std": "(°C)",  "VPD_std": "(kPa)",
}


def diagnose_area(area):
    print(f"\n--- {area.upper()} ---")
    d = os.path.join(DATA_DIR, area)

    # load artefacts
    imp_df       = pd.read_csv(os.path.join(d, f"permutation_importance_{area}.csv"))
    best_model   = joblib.load(os.path.join(d, f"best_model_{area}.joblib"))
    shap_vals    = np.load(os.path.join(d, f"shap_values_{area}.npy"))
    base_value   = np.load(os.path.join(d, f"shap_base_value_{area}.npy"))
    ice_data     = joblib.load(os.path.join(d, f"ice_data_{area}.joblib"))
    X_ice        = np.load(os.path.join(d, f"X_ice_sample_{area}.npy"))
    X            = np.load(os.path.join(d, f"X_{area}.npy"))
    y            = np.load(os.path.join(d, f"y_{area}.npy"))

    # --- permutation importance ---
    fig, ax = plt.subplots(figsize=(10, 8), dpi=DPI)
    y_pos = np.arange(len(imp_df))
    ax.barh(y_pos, imp_df["Importance"], xerr=imp_df["Std"],
            alpha=0.7, color="grey", ecolor="black", capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(imp_df["Feature"], fontsize=12)
    ax.invert_yaxis()
    ax.set_xlabel("Permutation Importance (ΔR²)", fontsize=12)
    ax.set_title(f"Feature Importance — {area.capitalize()}", fontsize=14,
                 fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(d, f"permutation_importance_{area}.pdf"), dpi=DPI)
    plt.close()

    # --- SHAP violin ---
    feat_imp = np.abs(shap_vals).mean(axis=0)
    sorted_idx = np.argsort(feat_imp)[::-1]
    plt.figure(figsize=(10, 8), dpi=DPI)
    shap.plots.violin(
        shap_vals[:, sorted_idx],
        features=X[:, sorted_idx],
        feature_names=[feature_names[i] for i in sorted_idx],
        plot_type="layered_violin",
        show=False,
    )
    plt.xlabel("SHAP value", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(d, f"shap_violin_{area}.pdf"), dpi=DPI)
    plt.close()

    # --- SHAP bar ---
    shap_imp_df = pd.DataFrame({
        "Feature":    feature_names,
        "Importance": feat_imp,
    }).sort_values("Importance")
    fig, ax = plt.subplots(figsize=(10, 8), dpi=DPI)
    ax.barh(range(len(shap_imp_df)), shap_imp_df["Importance"],
            alpha=0.7, color="grey")
    ax.set_yticks(range(len(shap_imp_df)))
    ax.set_yticklabels(shap_imp_df["Feature"], fontsize=12)
    ax.set_xlabel("mean(|SHAP value|)", fontsize=12)
    ax.set_title(f"SHAP Feature Importance — {area.capitalize()}", fontsize=14,
                 fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(d, f"shap_bar_{area}.pdf"), dpi=DPI)
    plt.close()

    # --- SHAP clustered bar (correlation-based) ---
    expl = shap.Explanation(
        values=shap_vals, base_values=base_value, data=X,
        feature_names=feature_names,
    )
    clustering = shap.utils.hclust(X, y, metric="correlation")
    fig = plt.figure(figsize=(10, 8), dpi=DPI)
    shap.plots.bar(expl, clustering=clustering,
                   max_display=len(feature_names), show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(d, f"shap_clustered_{area}.pdf"), dpi=DPI)
    plt.close()

    # --- ICE / PDP ---
    n_features = min(12, X.shape[1])
    fig, axes  = plt.subplots(4, 3, figsize=(15, 20), dpi=DPI)
    axes       = axes.flatten()

    for feat_idx in range(n_features):
        ax = axes[feat_idx]
        grid   = ice_data[feat_idx]["grid_values"]
        curves = ice_data[feat_idx]["individual"]
        pdp    = ice_data[feat_idx]["average"]

        for curve in curves:
            ax.plot(grid, curve, color="grey", alpha=0.05, linewidth=0.5)
        ax.plot(grid, pdp, color="black", linewidth=2.5, label="PDP")

        # density rug at the bottom (10% of plot height)
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        counts, bin_edges = np.histogram(
            X_ice[:, feat_idx], bins=50,
            range=(grid.min(), grid.max()),
        )
        counts_scaled = y_min + (counts / counts.max()) * (y_range * 0.1)
        ax.fill_between(bin_edges[:-1], y_min, counts_scaled,
                        step="post", alpha=0.3, color="gray",
                        edgecolor="black", linewidth=0.5)

        fname_label = feature_names[feat_idx]
        unit = UNITS.get(fname_label, "")
        ax.set_xlabel(f"{fname_label} {unit}".strip(), fontsize=10)
        ax.set_ylabel("Predicted TC (%)", fontsize=10)
        ax.set_ylim(y_min, y_max)
        ax.grid(axis="x", alpha=0.3)
        if feat_idx == 0:
            ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(d, f"ice_plots_{area}.pdf"), dpi=DPI)
    plt.close()
    print(f"  Saved diagnostic plots to {d}/")


for area in AREAS:
    diagnose_area(area)

print("\nDone.")
