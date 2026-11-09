"""
Multi-Objective Pareto Explorer
=================================
Explores the Pareto frontier of polymer chemical space for pairs of
(possibly conflicting) properties.

Methods:
  1. Weighted-sum scalarization over the known dataset
  2. NSGA-II-style non-dominated sorting on GA candidates
  3. Grid search over property space to map achievable combinations

Outputs:
  results/pareto_{propA}_vs_{propB}.csv   -- Pareto-optimal candidates
  results/pareto_coverage.csv             -- what property combinations are achievable
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations

POLY_DESIGN = Path(__file__).parent.parent
sys.path.insert(0, str(POLY_DESIGN))
sys.path.insert(0, str(Path(__file__).parent))

from data.loaders       import load_config, load_polyone, split_polyone
from predictor_oracle   import PredictorOracle, _is_pareto_efficient
from synthesizability   import filter_candidates, score_candidates

RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)


# ── Pareto front computation ──────────────────────────────────────────────────────

def compute_pareto_front(smiles_list: list[str],
                          oracle: PredictorOracle,
                          prop_a: str,
                          prop_b: str,
                          direction_a: str = "max",
                          direction_b: str = "max") -> pd.DataFrame:
    """
    Compute Pareto-optimal candidates for two objectives.
    direction: "max" or "min"
    """
    mean, std = oracle.predict(smiles_list)
    prop_names = oracle.prop_names

    if prop_a not in prop_names or prop_b not in prop_names:
        print(f"  Properties {prop_a} or {prop_b} not found.")
        return pd.DataFrame()

    idx_a = prop_names.index(prop_a)
    idx_b = prop_names.index(prop_b)

    vals_a = mean[:, idx_a] * (1 if direction_a == "max" else -1)
    vals_b = mean[:, idx_b] * (1 if direction_b == "max" else -1)

    costs        = np.column_stack([vals_a, vals_b])
    pareto_mask  = _is_pareto_efficient(costs)

    df = pd.DataFrame({
        "smiles":        smiles_list,
        f"pred_{prop_a}": mean[:, idx_a],
        f"pred_{prop_b}": mean[:, idx_b],
        f"unc_{prop_a}":  std[:, idx_a],
        f"unc_{prop_b}":  std[:, idx_b],
        "pareto_optimal": pareto_mask,
    })
    return df.sort_values(f"pred_{prop_a}", ascending=(direction_a == "min"))


def nsga2_sort(candidates_df: pd.DataFrame,
               objectives: list[tuple[str, str]]) -> pd.DataFrame:
    """
    Apply NSGA-II non-dominated sorting to a DataFrame of candidates.
    objectives: list of (column_name, "max"/"min")
    Returns DataFrame with added 'pareto_rank' and 'crowding_dist' columns.
    """
    obj_cols = []
    for col, direction in objectives:
        if col in candidates_df.columns:
            sign = 1 if direction == "max" else -1
            candidates_df[f"_obj_{col}"] = candidates_df[col] * sign
            obj_cols.append(f"_obj_{col}")

    if not obj_cols:
        return candidates_df

    obj_vals   = candidates_df[obj_cols].values
    n          = len(obj_vals)
    ranks      = np.zeros(n, dtype=int)
    remaining  = set(range(n))
    rank       = 0

    while remaining:
        front = []
        for i in remaining:
            dominated = False
            for j in remaining:
                if i == j:
                    continue
                if np.all(obj_vals[j] >= obj_vals[i]) and \
                   np.any(obj_vals[j] > obj_vals[i]):
                    dominated = True
                    break
            if not dominated:
                front.append(i)
        for i in front:
            ranks[i] = rank
            remaining.discard(i)
        rank += 1

    candidates_df = candidates_df.drop(columns=obj_cols)
    candidates_df["pareto_rank"] = ranks
    candidates_df["crowding_dist"] = _crowding_distance(obj_vals, ranks)
    return candidates_df.sort_values(["pareto_rank", "crowding_dist"],
                                      ascending=[True, False])


def _crowding_distance(obj_vals: np.ndarray, ranks: np.ndarray) -> np.ndarray:
    """Compute crowding distance for NSGA-II."""
    n, m    = obj_vals.shape
    dist    = np.zeros(n)
    for rank in np.unique(ranks):
        idx = np.where(ranks == rank)[0]
        if len(idx) <= 2:
            dist[idx] = np.inf
            continue
        for k in range(m):
            sorted_idx = idx[np.argsort(obj_vals[idx, k])]
            dist[sorted_idx[0]]  = np.inf
            dist[sorted_idx[-1]] = np.inf
            val_range = obj_vals[sorted_idx[-1], k] - obj_vals[sorted_idx[0], k]
            if val_range == 0:
                continue
            for i in range(1, len(sorted_idx) - 1):
                dist[sorted_idx[i]] += (
                    obj_vals[sorted_idx[i+1], k] - obj_vals[sorted_idx[i-1], k]
                ) / val_range
    return dist


# ── Dataset-level Pareto analysis ────────────────────────────────────────────────

def pareto_from_dataset(oracle: PredictorOracle,
                         prop_pairs: list[tuple[str, str, str, str]],
                         n_samples: int = 500,
                         cfg=None) -> dict[str, pd.DataFrame]:
    """
    Compute Pareto fronts for multiple property pairs on a dataset subset.
    prop_pairs: [(prop_a, dir_a, prop_b, dir_b), ...]
    """
    if cfg is None:
        cfg = load_config()
    df   = load_polyone(cfg)
    sub  = df.sample(min(n_samples, len(df)), random_state=42)
    smi  = sub["smiles"].tolist()

    results = {}
    for prop_a, dir_a, prop_b, dir_b in prop_pairs:
        key = f"{prop_a}_vs_{prop_b}"
        print(f"\n── Pareto: {prop_a} ({dir_a}) vs {prop_b} ({dir_b}) ──")

        pareto_df = compute_pareto_front(smi, oracle, prop_a, prop_b, dir_a, dir_b)
        if pareto_df.empty:
            continue

        # Filter to valid and synthesizable
        valid_smi = pareto_df["smiles"].tolist()
        filt      = filter_candidates(valid_smi)
        pareto_df = pareto_df.merge(
            filt[["smiles", "sa_score", "passes_filter"]], on="smiles", how="left"
        )

        n_pareto = pareto_df["pareto_optimal"].sum()
        print(f"  Total: {len(pareto_df)}  Pareto-optimal: {n_pareto}")

        out_path = RESULTS / f"pareto_{key}.csv"
        pareto_df.to_csv(out_path, index=False)
        print(f"  Saved → {out_path}")
        results[key] = pareto_df

    return results


# ── Entry point ──────────────────────────────────────────────────────────────────

def run_pareto_explorer():
    cfg    = load_config()
    oracle = PredictorOracle.from_checkpoint(cfg)

    prop_pairs = [
        ("Tg",  "max", "Egb", "min"),    # high Tg + low bandgap (optoelectronics)
        ("Tg",  "max", "Td",  "max"),    # thermal stability
        ("Egb", "min", "rho", "min"),    # light + semiconducting
        ("Tg",  "max", "Cp",  "max"),    # thermal management
    ]

    results = pareto_from_dataset(oracle, prop_pairs, n_samples=500, cfg=cfg)

    # NSGA-II ranking on combined candidates
    if results:
        all_candidates = pd.concat(list(results.values()))
        all_candidates = all_candidates.drop_duplicates("smiles")
        ranked = nsga2_sort(
            all_candidates,
            objectives=[("pred_Tg", "max"), ("pred_Egb", "min")]
        )
        ranked.to_csv(RESULTS / "pareto_nsga2_ranked.csv", index=False)
        print(f"\nNSGA-II ranked saved → {RESULTS / 'pareto_nsga2_ranked.csv'}")

    return results


if __name__ == "__main__":
    run_pareto_explorer()
