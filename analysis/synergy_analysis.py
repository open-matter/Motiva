from __future__ import annotations
"""
Non-linear Functional Group Synergy Analysis
=============================================
For each pair of FGs (i, j) that co-occur in polymers, tests whether their
combined effect on polymer properties is additive (as VK assumes) or
synergistic/antagonistic:

  synergy(i,j,P) = effect(i+j, P) - effect(i, P) - effect(j, P)
                                                   â†?additive expectation

A positive synergy score means the pair amplifies the property beyond
what each individual FG contributes.  A negative score means antagonism.

This analysis is the key scientific novelty beyond van Krevelen.

Method:
  Uses LightGBM with pairwise FG interaction features and SHAP interaction
  values, which directly decompose predictions into main effects + interactions.

Outputs:
  results/fg_synergy_matrix_{property}.csv  -- (FGÃ—FG) synergy scores
  results/fg_synergy_top20.csv              -- top synergistic pairs
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations

POLY_DESIGN = Path(__file__).parent.parent
sys.path.insert(0, str(POLY_DESIGN))

from data.loaders  import load_config, load_polyone, split_polyone, get_mask
from fg_extractor  import batch_smiles_to_fg_binary, get_fg_names, get_fg_metadata

RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)


# â”€â”€ Method 1: SHAP Interaction Values â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def shap_interaction_analysis(
    X_train: np.ndarray,
    y_train: np.ndarray,
    mask_train: np.ndarray,
    X_test: np.ndarray,
    fg_names: list[str],
    prop_names: list[str],
    top_k: int = 20,
    max_features: int = 40,
) -> dict[str, pd.DataFrame]:
    """
    Returns dict {property: synergy_matrix_df} using SHAP interaction values.
    """
    try:
        import lightgbm as lgb
        import shap
    except ImportError:
        print("[synergy] lightgbm or shap not installed.")
        return {}

    results = {}
    # Use only FGs present in â‰?% of samples (reduce noise)
    freq = X_train.mean(axis=0)
    keep = np.where(freq >= 0.05)[0][:max_features]
    fg_subset = [fg_names[i] for i in keep]
    X_tr_sub  = X_train[:, keep]
    X_te_sub  = X_test[:,  keep]

    for p_idx, prop in enumerate(prop_names):
        valid = mask_train[:, p_idx]
        if valid.sum() < 20:
            continue
        print(f"  Computing SHAP interactions for {prop} ...")

        model = lgb.LGBMRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, n_jobs=-1, random_state=42, verbose=-1
        )
        model.fit(X_tr_sub[valid], y_train[valid, p_idx])

        explainer  = shap.TreeExplainer(model)
        # interaction_values shape: (N_test, n_fg, n_fg)
        inter_vals = explainer.shap_interaction_values(X_te_sub)
        # Mean absolute interaction across test samples
        mean_inter = np.abs(inter_vals).mean(axis=0)           # (n_fg, n_fg)
        # Zero out diagonal (main effects)
        np.fill_diagonal(mean_inter, 0.0)

        synergy_df = pd.DataFrame(mean_inter, index=fg_subset, columns=fg_subset)
        results[prop] = synergy_df
        synergy_df.to_csv(RESULTS / f"fg_synergy_matrix_{prop}.csv")
        print(f"    Saved synergy matrix â†?{RESULTS / f'fg_synergy_matrix_{prop}.csv'}")

    return results


# â”€â”€ Method 2: Empirical Co-occurrence Synergy â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def empirical_synergy(
    X: np.ndarray,
    y: np.ndarray,
    masks: np.ndarray,
    fg_names: list[str],
    prop_names: list[str],
    min_count: int = 10,
) -> pd.DataFrame:
    """
    Empirical test: for each FG pair (i,j), compare:
      mean(Y | i=1, j=1) vs mean(Y | i=1) + mean(Y | j=1) - mean(Y)
    Returns DataFrame with columns: fg_i, fg_j, property, synergy_score, n_ij.
    """
    rows = []
    global_mean = np.nanmean(np.where(masks, y, np.nan), axis=0)  # (n_props,)

    fg_pairs = list(combinations(range(len(fg_names)), 2))
    total = len(fg_pairs) * len(prop_names)
    done  = 0

    for i, j in fg_pairs:
        mask_i   = X[:, i] > 0
        mask_j   = X[:, j] > 0
        mask_ij  = mask_i & mask_j
        if mask_ij.sum() < min_count:
            continue

        for p_idx, prop in enumerate(prop_names):
            valid = masks[:, p_idx]
            if (valid & mask_ij).sum() < min_count:
                continue

            def mean_valid(m):
                sel = valid & m
                return float(np.nanmean(y[sel, p_idx])) if sel.sum() > 0 else np.nan

            mu_i  = mean_valid(mask_i)
            mu_j  = mean_valid(mask_j)
            mu_ij = mean_valid(mask_ij)
            mu_g  = global_mean[p_idx]

            if any(np.isnan([mu_i, mu_j, mu_ij, mu_g])):
                continue

            additive_pred = mu_i + mu_j - mu_g
            synergy = mu_ij - additive_pred
            rows.append({
                "fg_i":         fg_names[i],
                "fg_j":         fg_names[j],
                "property":     prop,
                "mu_i":         round(mu_i, 4),
                "mu_j":         round(mu_j, 4),
                "mu_ij":        round(mu_ij, 4),
                "additive_pred":round(additive_pred, 4),
                "synergy_score":round(synergy, 4),
                "n_ij":         int(mask_ij.sum()),
            })
        done += len(prop_names)
        if done % 5000 == 0:
            print(f"  Pairs processed: {done}/{total}", end="\r")

    return pd.DataFrame(rows)


# â”€â”€ Top synergy pairs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def get_top_synergy_pairs(synergy_df: pd.DataFrame,
                           top_k: int = 20,
                           prop_filter: str | None = None) -> pd.DataFrame:
    df = synergy_df.copy()
    if prop_filter:
        df = df[df["property"] == prop_filter]
    df["abs_synergy"] = df["synergy_score"].abs()
    return df.sort_values("abs_synergy", ascending=False).head(top_k)


# â”€â”€ Entry point â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def run_synergy_analysis():
    cfg   = load_config()
    df    = load_polyone(cfg)
    props = cfg["properties"]["polyone_columns"]
    names = cfg["properties"]["names"]

    train_df, _, test_df = split_polyone(df, cfg)
    smiles_all = df["smiles"].tolist()

    from data.loaders import fit_scalers, apply_scalers
    scalers  = fit_scalers(train_df, props)
    y_all    = apply_scalers(df, scalers, props)
    m_all    = get_mask(df, props)

    fg_names = get_fg_names(cfg)
    X_all    = batch_smiles_to_fg_binary(smiles_all, cfg)

    # 1. Empirical synergy
    print("\nâ”€â”€ Empirical Co-occurrence Synergy â”€â”€")
    syn_df = empirical_synergy(X_all, y_all, m_all, fg_names, names)
    if not syn_df.empty:
        syn_df.to_csv(RESULTS / "fg_synergy_empirical.csv", index=False)
        top = get_top_synergy_pairs(syn_df, top_k=20)
        print(f"\n  Top 20 synergistic FG pairs:")
        print(top[["fg_i", "fg_j", "property", "synergy_score", "n_ij"]].to_string(index=False))
        top.to_csv(RESULTS / "fg_synergy_top20.csv", index=False)
        print(f"\n  Saved â†?{RESULTS / 'fg_synergy_empirical.csv'}")

    # 2. SHAP interaction values
    print("\nâ”€â”€ SHAP Interaction Values â”€â”€")
    smiles_tr = train_df["smiles"].tolist()
    smiles_te = test_df["smiles"].tolist()
    X_tr = batch_smiles_to_fg_binary(smiles_tr, cfg)
    X_te = batch_smiles_to_fg_binary(smiles_te, cfg)
    from data.loaders import apply_scalers
    y_tr = apply_scalers(train_df, scalers, props)
    m_tr = get_mask(train_df, props)

    shap_results = shap_interaction_analysis(X_tr, y_tr, m_tr, X_te,
                                              fg_names, names)
    for prop, mat in shap_results.items():
        stacked = mat.stack().reset_index()
        stacked.columns = ["fg_i", "fg_j", "shap_interaction"]
        stacked = stacked[stacked["fg_i"] != stacked["fg_j"]]
        stacked = stacked.sort_values("shap_interaction", ascending=False)
        print(f"\n  Top SHAP interactions for {prop}:")
        print(stacked.head(10).to_string(index=False))

    return syn_df


if __name__ == "__main__":
    run_synergy_analysis()
