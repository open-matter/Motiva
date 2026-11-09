"""
Attribution Pipeline Master Runner
====================================
Runs all attribution analyses in sequence:
  1. FG feature extraction
  2. SHAP attribution (LightGBM-based)
  3. Perturbation attribution (RoBERTa model-based, if available)
  4. Van Krevelen comparison
  5. Synergy analysis

Usage:
  python run_attribution.py
  python run_attribution.py --skip_model   # skip model perturbation
  python run_attribution.py --prop Tg Egb  # analyze specific properties
"""
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

POLY_DESIGN = Path(__file__).parent.parent
sys.path.insert(0, str(POLY_DESIGN))
sys.path.insert(0, str(Path(__file__).parent))

from data.loaders       import load_config, load_polyone, split_polyone, get_mask
from fg_extractor       import batch_smiles_to_fg_binary, get_fg_names, fg_counts_in_dataset
from fg_perturbation    import run_attribution
from van_krevelen       import run_vk_comparison
from synergy_analysis   import run_synergy_analysis

RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)


def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_model", action="store_true",
                        help="Skip model perturbation (only SHAP)")
    parser.add_argument("--prop", nargs="+", default=["Tg", "Egb"],
                        help="Properties for VK comparison")
    args = parser.parse_args()

    cfg = load_config()

    # ── Step 1: FG frequency analysis ──────────────────────────────────────────
    print_section("Step 1: FG Frequency Analysis")
    df = load_polyone(cfg)
    smiles = df["smiles"].tolist()
    counts = fg_counts_in_dataset(smiles, cfg)
    print(f"  Dataset: {len(smiles)} polymers")
    print(f"\n  Top 20 most common FGs:")
    print(counts.head(20).to_string())
    counts.to_csv(RESULTS / "fg_frequency.csv", header=["count"])
    print(f"\n  Saved → {RESULTS / 'fg_frequency.csv'}")

    # ── Step 2: SHAP + Perturbation Attribution ─────────────────────────────────
    print_section("Step 2: Attribution Analysis (SHAP + Perturbation)")
    shap_df, pert_df = run_attribution(
        use_model_perturbation=not args.skip_model
    )

    if shap_df is not None and not shap_df.empty:
        print("\n  SHAP attribution summary (top FGs by Tg):")
        if "Tg" in shap_df.columns:
            top_tg = shap_df["Tg"].abs().sort_values(ascending=False).head(10)
            print(top_tg.to_string())

    # ── Step 3: Van Krevelen Comparison ─────────────────────────────────────────
    print_section("Step 3: Van Krevelen Comparison")
    for prop in args.prop:
        run_vk_comparison(
            ml_attribution_path=RESULTS / "fg_attribution_shap.csv",
            property_name=prop,
        )

    # ── Step 4: Synergy Analysis ─────────────────────────────────────────────────
    print_section("Step 4: Non-linear Synergy Analysis")
    syn_df = run_synergy_analysis()

    # ── Step 5: Summary report ───────────────────────────────────────────────────
    print_section("Summary")
    print("  All attribution results saved in 2_attribution/results/")
    print("  Files generated:")
    for f in sorted(RESULTS.iterdir()):
        print(f"    {f.name}")


if __name__ == "__main__":
    main()
