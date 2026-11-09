"""
Inverse Design Master Runner
==============================
Coordinates all inverse design modules:
  1. Load oracle (trained predictor)
  2. Run genetic algorithm toward specified property targets
  3. Filter by synthesizability
  4. Compute Pareto frontiers for property pairs
  5. Save top candidates for DFT validation

Usage:
  python run_inverse_design.py
  python run_inverse_design.py --targets Tg:400:600 Egb:1.5:3.0
  python run_inverse_design.py --only_pareto
"""
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

POLY_DESIGN = Path(__file__).parent.parent
sys.path.insert(0, str(POLY_DESIGN))
sys.path.insert(0, str(Path(__file__).parent))

from data.loaders        import load_config, load_polyone
from predictor_oracle    import PredictorOracle
from genetic_optimizer   import GeneticOptimizer
from pareto_explorer     import run_pareto_explorer, compute_pareto_front
from synthesizability    import filter_candidates, score_candidates

RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)


def parse_targets(target_strs: list[str]) -> dict[str, tuple[float, float]]:
    """Parse 'Prop:lo:hi' strings into dict."""
    targets = {}
    for s in target_strs:
        parts = s.split(":")
        if len(parts) == 3:
            targets[parts[0]] = (float(parts[1]), float(parts[2]))
    return targets


def run_genetic_design(oracle, seed_smiles, targets, cfg):
    trcfg = cfg["inverse_design"]
    ga = GeneticOptimizer(
        oracle,
        population_size = trcfg["population_size"],
        n_generations   = trcfg["n_generations"],
        mutation_rate   = trcfg["mutation_rate"],
        seed            = cfg["training"]["seed"],
    )
    print(f"\n  GA targets: {targets}")
    ga_results = ga.run(seed_smiles, targets, verbose=True)
    return ga_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", nargs="+",
                        default=["Tg:400:600", "Egb:1.5:3.0"],
                        help="Property targets in Prop:lo:hi format")
    parser.add_argument("--only_pareto", action="store_true")
    parser.add_argument("--n_seed", type=int, default=200,
                        help="Number of seed polymers from dataset")
    args = parser.parse_args()

    cfg     = load_config()
    targets = parse_targets(args.targets)
    print(f"Targets: {targets}")

    # ── Load oracle ──────────────────────────────────────────────────────────────
    print("\n── Loading Predictor Oracle ──")
    oracle = PredictorOracle.from_checkpoint(cfg)

    # ── Load seed dataset ────────────────────────────────────────────────────────
    df           = load_polyone(cfg)
    seed_smiles  = df["smiles"].sample(
        min(args.n_seed, len(df)), random_state=cfg["training"]["seed"]
    ).tolist()
    print(f"  Seed pool: {len(seed_smiles)} polymers")

    # ── Screen existing dataset first ────────────────────────────────────────────
    print("\n── Screening Existing Dataset ──")
    mean, std = oracle.predict(seed_smiles)
    fitness   = oracle.fitness(seed_smiles, targets)
    top_idx   = np.argsort(fitness)[-50:][::-1]
    top_smi   = [seed_smiles[i] for i in top_idx]
    top_scores= fitness[top_idx]
    print(f"  Top 10 existing polymers by fitness:")
    for smi, sc in zip(top_smi[:10], top_scores[:10]):
        print(f"    {sc:.4f}  {smi[:60]}")

    # ── Genetic Algorithm ────────────────────────────────────────────────────────
    if not args.only_pareto:
        print("\n── Genetic Algorithm Inverse Design ──")
        ga_results = run_genetic_design(oracle, seed_smiles, targets, cfg)

        # Filter by synthesizability
        all_smiles = ga_results["smiles"].tolist()
        pred_mean, _ = oracle.predict(all_smiles)
        filtered_df  = filter_candidates(all_smiles, predictions=pred_mean)
        filtered_df  = score_candidates(filtered_df, targets)

        passed = filtered_df[filtered_df["passes_filter"]]
        print(f"\n  Valid candidates: {len(passed)} / {len(filtered_df)}")
        passed.to_csv(RESULTS / "inverse_design_candidates.csv", index=False)
        print(f"  Saved → {RESULTS / 'inverse_design_candidates.csv'}")

        # Combine existing top + GA candidates
        combined_smi = top_smi + passed.head(50)["smiles"].tolist()
        combined_smi = list(set(combined_smi))
        pred_c, std_c = oracle.predict(combined_smi)
        all_cands_df  = filter_candidates(combined_smi, predictions=pred_c)
        all_cands_df  = score_candidates(all_cands_df, targets)
        all_cands_df.to_csv(RESULTS / "all_candidates_ranked.csv", index=False)
        print(f"  All candidates ranked → {RESULTS / 'all_candidates_ranked.csv'}")

    # ── Pareto Analysis ──────────────────────────────────────────────────────────
    print("\n── Pareto Frontier Analysis ──")
    run_pareto_explorer()

    # ── Summary ──────────────────────────────────────────────────────────────────
    print("\n── Summary ──")
    print("  Files saved:")
    for f in sorted(RESULTS.iterdir()):
        print(f"    {f.name}")


if __name__ == "__main__":
    main()
