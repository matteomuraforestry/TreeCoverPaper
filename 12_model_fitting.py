"""
13_model_fitting.py

Fit HistGradientBoostingRegressor models for all five areas
(global, tropics, arid, temperate, boreal) using Bayesian hyperparameter
optimisation (scikit-optimize BayesSearchCV).

Cross-validation uses RepeatedKFold (3 splits × 5 repeats = 15 folds) with
R², RMSE, and MAE as scoring metrics.

After fitting, the script computes:
  - permutation feature importance (30 repeats)
  - SHAP values (TreeExplainer)
  - ICE / PDP curves for all predictors

All outputs are saved per area in DATA_DIR/{area}/.

Requirements: scikit-learn, scikit-optimize, shap, joblib

Inputs:
    X_{area}.npy, y_{area}.npy  (one pair per area)
    names.npy                   (feature names, band 0 = TC label)
Outputs per area:
    best_model_{area}.joblib
    bayes_search_cv_results_{area}.csv
    permutation_importance_{area}.csv
    shap_values_{area}.npy, shap_base_value_{area}.npy
    ice_data_{area}.joblib
"""

import os
import numpy as np
import pandas as pd
import joblib
import shap

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import RepeatedKFold, cross_validate
from sklearn.inspection import permutation_importance, partial_dependence
from skopt import BayesSearchCV
from skopt.space import Real, Integer

DATA_DIR = r""
SEED     = 86
AREAS    = ["arid", "tropics", "temperate", "boreal", "global"]

feature_names = np.load(os.path.join(DATA_DIR, "names.npy"))[1:]
print("Features:", feature_names)


def fit_area(area):
    print(f"\n{'='*60}\nFitting model for: {area.upper()}\n{'='*60}")
    area_dir = os.path.join(DATA_DIR, area)
    os.makedirs(area_dir, exist_ok=True)

    X = np.load(os.path.join(area_dir, f"X_{area}.npy"))
    y = np.load(os.path.join(area_dir, f"y_{area}.npy"))
    print(f"  X={X.shape}  y={y.shape}")

    # base model — last feature (year) treated as categorical
    base_model = HistGradientBoostingRegressor(
        random_state=SEED,
        categorical_features=[X.shape[1] - 1],
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
    )

    param_space = {
        "learning_rate": Real(0.05, 0.3, prior="uniform"),
        "max_iter":      Integer(100, 300),
    }

    rkf = RepeatedKFold(n_splits=3, n_repeats=5, random_state=SEED)

    bayes = BayesSearchCV(
        estimator=base_model,
        search_spaces=param_space,
        n_iter=30,
        cv=rkf,
        n_jobs=-1,
        scoring="r2",
        verbose=0,
        random_state=SEED,
        return_train_score=False,
    )
    bayes.fit(X, y)
    joblib.dump(bayes, os.path.join(area_dir, f"bayes_search_{area}.joblib"))

    best_model = bayes.best_estimator_
    print(f"  Best params: {bayes.best_params_}")
    print(f"  Best CV R²:  {bayes.best_score_:.4f}")
    joblib.dump(best_model, os.path.join(area_dir, f"best_model_{area}.joblib"))

    cv_df = pd.DataFrame(bayes.cv_results_)
    cv_df.to_csv(os.path.join(area_dir, f"bayes_search_cv_results_{area}.csv"),
                 index=False)

    # cross-validate best parameters for R²/RMSE/MAE with uncertainty
    model_eval = HistGradientBoostingRegressor(
        **bayes.best_params_,
        random_state=SEED,
        categorical_features=[X.shape[1] - 1],
    )
    cv_scores = cross_validate(
        model_eval, X, y,
        cv=RepeatedKFold(n_splits=3, n_repeats=5, random_state=SEED),
        scoring=["r2", "neg_root_mean_squared_error", "neg_mean_absolute_error"],
        n_jobs=-1,
    )
    r2   = cv_scores["test_r2"]
    rmse = -cv_scores["test_neg_root_mean_squared_error"]
    mae  = -cv_scores["test_neg_mean_absolute_error"]
    print(f"  R²:   {r2.mean():.4f} ± {r2.std():.4f}")
    print(f"  RMSE: {rmse.mean():.4f} ± {rmse.std():.4f}")
    print(f"  MAE:  {mae.mean():.4f} ± {mae.std():.4f}")

    # permutation importance
    perm = permutation_importance(
        best_model, X, y,
        n_repeats=30,
        random_state=SEED,
        n_jobs=-1,
        scoring="r2",
    )
    imp_df = pd.DataFrame({
        "Feature":    feature_names,
        "Importance": perm.importances_mean,
        "Std":        perm.importances_std,
    }).sort_values("Importance", ascending=False)
    imp_df.to_csv(
        os.path.join(area_dir, f"permutation_importance_{area}.csv"), index=False
    )
    print(f"  Top feature: {imp_df.iloc[0]['Feature']} "
          f"({imp_df.iloc[0]['Importance']:.4f})")

    # ICE / PDP for all features
    n_ice = min(5000, X.shape[0])
    ice_idx = np.random.choice(X.shape[0], n_ice, replace=False)
    X_ice   = X[ice_idx]
    ice_data = {}
    for feat_idx in range(X.shape[1]):
        pd_res = partial_dependence(
            best_model, X_ice,
            features=[feat_idx],
            kind="both",
            grid_resolution=50,
        )
        ice_data[feat_idx] = {
            "grid_values": pd_res["grid_values"][0],
            "individual":  pd_res["individual"][0],
            "average":     pd_res["average"][0],
        }
    joblib.dump(ice_data, os.path.join(area_dir, f"ice_data_{area}.joblib"))
    np.save(os.path.join(area_dir, f"ice_sample_indices_{area}.npy"), ice_idx)
    np.save(os.path.join(area_dir, f"X_ice_sample_{area}.npy"),       X_ice)

    # SHAP values
    explainer   = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X)
    np.save(os.path.join(area_dir, f"shap_values_{area}.npy"),    shap_values)
    np.save(os.path.join(area_dir, f"shap_base_value_{area}.npy"), explainer.expected_value)
    print(f"  SHAP values shape: {shap_values.shape}")


for area in AREAS:
    fit_area(area)

print("\nAll areas processed.")
