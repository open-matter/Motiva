from __future__ import annotations
"""
Van Krevelen Group Contribution Comparison
===========================================
Compares ML-learned FG contributions (from perturbation/SHAP analysis)
with classical van Krevelen (VK) group contribution theory.

Key analyses:
  1. Rank correlation between ML attribution and VK Tg contributions
  2. Identify FGs where ML diverges from VK (novel insights)
  3. Quantify how much each FG explains variance (ML vs VK)
  4. Non-linear synergy detection (two-FG combinations)

Outputs:
  results/vk_comparison.csv       -- side-by-side ML vs VK contributions
  results/vk_divergence.csv       -- top divergent FGs (ML insight beyond VK)
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr, pearsonr

POLY_DESIGN = Path(__file__).parent.parent
sys.path.insert(0, str(POLY_DESIGN))

VK_DATA = Path(__file__).parent / "van_krevelen_data.csv"
RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)


# â”€â”€ Load VK data â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def load_vk_data() -> pd.DataFrame:
    """Load van Krevelen group contribution table."""
    df = pd.read_csv(VK_DATA)
    # Compute simplified VK Tg contribution = Yg / V
    df["Tg_vk_contrib"] = df["Tg_Yg"] / df["Tg_V"].replace(0, np.nan)
    return df


def vk_predict_tg(smiles_list: list[str], vk_df: pd.DataFrame,
                   fg_binary: np.ndarray, fg_names: list[str]) -> np.ndarray:
    """
    Estimate Tg using additive VK group contributions.
    Tg â‰?Î£(Y_g_i * count_i) / Î£(V_i * count_i)
    Returns array of predicted Tg (K).  NaN if no matched FGs.
    """
    # Build VK lookup by FG name
    vk_lookup = {}
    for _, row in vk_df.iterrows():
        vk_lookup[row["fg_name"]] = {
            "Yg": row["Tg_Yg"],
            "V":  row["Tg_V"],
        }

    tg_preds = np.full(len(smiles_list), np.nan)
    for i in range(len(smiles_list)):
        total_yg = 0.0
        total_v  = 0.0
        has_match = False
        for j, name in enumerate(fg_names):
            if fg_binary[i, j] > 0 and name in vk_lookup:
                total_yg += vk_lookup[name]["Yg"]
                total_v  += vk_lookup[name]["V"]
                has_match = True
        if has_match and total_v > 0:
            tg_preds[i] = total_yg / total_v
    return tg_preds


# â”€â”€ Comparison â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def compare_ml_vs_vk(ml_attribution_df: pd.DataFrame,
                     vk_df: pd.DataFrame,
                     property_name: str = "Tg") -> pd.DataFrame:
    """
    Merge ML attribution and VK contributions for a specific property.
    Returns DataFrame with columns: fg_name, ml_contrib, vk_contrib,
                                     vk_direction, divergence_score, source.
    """
    # Build VK contribution series
    vk_series = vk_df.set_index("fg_name")["Tg_vk_contrib"]

    rows = []
    for fg in ml_attribution_df.index:
        ml_val = ml_attribution_df.loc[fg, property_name] \
                 if property_name in ml_attribution_df.columns else np.nan
        vk_val = vk_series.get(fg, np.nan)
        vk_note = ""
        if fg in vk_df["fg_name"].values:
            vk_note = vk_df.loc[vk_df["fg_name"] == fg, "Tg_note"].values[0]
        rows.append({
            "fg_name":    fg,
            "ml_contrib": ml_val,
            "vk_contrib": vk_val,
            "vk_note":    vk_note,
            "in_vk":      not np.isnan(vk_val),
        })
    df = pd.DataFrame(rows)

    # Divergence score: |sign(ml) - sign(vk)| or normalized difference
    valid = df["in_vk"] & df["ml_contrib"].notna() & df["vk_contrib"].notna()
    if valid.any():
        ml_norm = df.loc[valid, "ml_contrib"]
        vk_norm = df.loc[valid, "vk_contrib"]
        ml_norm = (ml_norm - ml_norm.mean()) / (ml_norm.std() + 1e-8)
        vk_norm = (vk_norm - vk_norm.mean()) / (vk_norm.std() + 1e-8)
        df.loc[valid, "divergence_score"] = (ml_norm - vk_norm).abs()
    else:
        df["divergence_score"] = np.nan

    return df.sort_values("divergence_score", ascending=False)


def correlation_analysis(compare_df: pd.DataFrame) -> dict:
    """Compute Spearman and Pearson correlation between ML and VK contributions."""
    valid = compare_df["in_vk"] & compare_df["ml_contrib"].notna()
    if valid.sum() < 3:
        return {"pearson_r": np.nan, "spearman_r": np.nan}

    ml = compare_df.loc[valid, "ml_contrib"].values
    vk = compare_df.loc[valid, "vk_contrib"].values
    rp, _ = pearsonr(ml,  vk)
    rs, _ = spearmanr(ml, vk)
    return {"pearson_r": round(rp, 4), "spearman_r": round(rs, 4),
            "n_shared": int(valid.sum())}


def novel_fgs(compare_df: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    """FGs with high ML attribution but NOT in VK database (novel discoveries)."""
    novel = compare_df[~compare_df["in_vk"]].copy()
    novel = novel.dropna(subset=["ml_contrib"])
    novel["ml_abs"] = novel["ml_contrib"].abs()
    return novel.sort_values("ml_abs", ascending=False).head(top_k)


def run_vk_comparison(ml_attribution_path: str | Path | None = None,
                       property_name: str = "Tg") -> pd.DataFrame:
    """
    Main entry: load ML attribution results and compare with VK.
    """
    print(f"\nâ”€â”€ Van Krevelen Comparison for {property_name} â”€â”€")

    # Load ML attribution
    if ml_attribution_path is None:
        ml_attribution_path = RESULTS / "fg_attribution_shap.csv"
    if not Path(ml_attribution_path).exists():
        print(f"  ML attribution file not found: {ml_attribution_path}")
        print("  Run 2_attribution/fg_perturbation.py first.")
        return pd.DataFrame()

    ml_df = pd.read_csv(ml_attribution_path, index_col=0)
    vk_df = load_vk_data()

    # Comparison
    compare_df = compare_ml_vs_vk(ml_df, vk_df, property_name)
    corr = correlation_analysis(compare_df)
    print(f"  ML vs VK Pearson r:  {corr['pearson_r']}")
    print(f"  ML vs VK Spearman r: {corr['spearman_r']}")
    print(f"  Shared FGs:          {corr['n_shared']}")

    # Novel FGs not covered by VK
    novel = novel_fgs(compare_df)
    print(f"\n  Top {len(novel)} novel FGs (not in VK, but high ML importance):")
    print(novel[["fg_name", "ml_contrib"]].to_string(index=False))

    # Divergent FGs (ML disagrees with VK direction)
    diver = compare_df[compare_df["in_vk"]].dropna(subset=["divergence_score"])
    print(f"\n  Top divergent FGs (ML vs VK):")
    print(diver[["fg_name", "ml_contrib", "vk_contrib", "divergence_score"]]
          .head(10).to_string(index=False))

    # Save
    compare_df.to_csv(RESULTS / f"vk_comparison_{property_name}.csv", index=False)
    print(f"\n  Saved â†?{RESULTS / f'vk_comparison_{property_name}.csv'}")
    return compare_df


if __name__ == "__main__":
    run_vk_comparison(property_name="Tg")
    run_vk_comparison(property_name="Egb")
