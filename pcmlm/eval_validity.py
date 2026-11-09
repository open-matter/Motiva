"""
Proper GROVE sequence validity checker.

GROVE encoding rules (from smiles2sum_0624.py):
  - ^ marks branch block boundaries: ^content^ = one branch block
  - ^^ between two blocks means sequential non-fused rings (close one, open next)
  - Every ^ must have a pair -> total count must be EVEN
  - ^^^ or more consecutive carets are structurally impossible
  - A string starting with ^^ is invalid (nothing to close before the first ^)
  - | only appears inside branch blocks (between the two ^ of a block)

Validity levels:
  L1 (format): non-empty string
  L2 (structural): L1 + even ^ count + no ^^^ + no leading ^^
  L3 (vocab): L2 + all tokens are in training GROVE vocabulary (no UNK)
"""
import csv
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def grove_structural_valid(s):
    """
    Check structural GROVE validity rules.
    Returns (is_valid, reason).
    """
    if not s or not s.strip():
        return False, "empty"

    caret_count = s.count('^')

    # Even ^ count: each opening ^ must have a closing ^
    if caret_count % 2 != 0:
        return False, f"odd_caret_{caret_count}"

    # No ^^^: at most ^^ is valid (sequential ring open/close)
    if '^^^' in s:
        return False, "triple_caret"

    # No leading ^^: would imply closing a block that was never opened
    if s.startswith('^^'):
        return False, "leading_double_caret"

    return True, "ok"


def evaluate_csv(csv_path, gen_col, train_csv_path, label):
    """Re-evaluate a generated results CSV with proper GROVE validity."""
    # Load training structures for novelty check
    train_set = set()
    with open(train_csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            train_set.add(row["structure"])

    results = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            results.append(row)

    N = len(results)
    gen = [r.get(gen_col, "") for r in results]

    # L1: format valid (non-empty)
    fmt_valid = [g for g in gen if g and g.strip()]

    # L2: structural valid
    struct_valid = [g for g in fmt_valid if grove_structural_valid(g)[0]]

    # Failure breakdown
    failures = {}
    for g in fmt_valid:
        ok, reason = grove_structural_valid(g)
        if not ok:
            failures[reason] = failures.get(reason, 0) + 1

    # Novelty on L2-valid set
    unique = list(set(struct_valid))
    novel  = [g for g in unique if g not in train_set]

    print(f"\n{'='*65}")
    print(f"Eval [{label}]  N={N}")
    print(f"  L1 format_valid  : {len(fmt_valid):3d} / {N}  ({len(fmt_valid)/N*100:.1f}%)")
    print(f"  L2 struct_valid  : {len(struct_valid):3d} / {N}  ({len(struct_valid)/N*100:.1f}%)")
    if failures:
        for reason, cnt in sorted(failures.items(), key=lambda x: -x[1]):
            print(f"      fail [{reason}]: {cnt}")
    print(f"  unique (of L2)   : {len(unique)}")
    if unique:
        print(f"  novel (of unique): {len(novel)} ({len(novel)/len(unique)*100:.1f}%)")
    print(f"{'='*65}")

    return {
        "label": label,
        "N": N,
        "L1_fmt": len(fmt_valid),
        "L1_pct": round(len(fmt_valid)/N*100, 1),
        "L2_struct": len(struct_valid),
        "L2_pct": round(len(struct_valid)/N*100, 1),
        "unique": len(unique),
        "novel": len(novel),
        "novelty_pct": round(len(novel)/len(unique)*100, 1) if unique else 0.0,
        "failures": failures,
    }


def show_failure_examples(csv_path, gen_col, n=5):
    """Print examples of structurally invalid sequences."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"\nInvalid examples from {os.path.basename(csv_path)}:")
    shown = 0
    for row in rows:
        g = row.get(gen_col, "")
        if not g or not g.strip():
            continue
        ok, reason = grove_structural_valid(g)
        if not ok:
            print(f"  [{reason}] true: {row['structure'][:45]}")
            print(f"           gen : {g[:60]}")
            shown += 1
            if shown >= n:
                break


if __name__ == "__main__":
    BASE     = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE, "data_1968AUG", "processed")
    TRAIN_CSV = os.path.join(BASE, "data_1968AUG", "processed", "data_split.csv")

    COND_CSV  = os.path.join(DATA_DIR, "gen_highmasked_cluster_cond.csv")
    UNCOND_CSV = os.path.join(DATA_DIR, "gen_highmasked_cluster_uncond.csv")

    print("="*65)
    print("GROVE Structural Validity Re-evaluation")
    print("Rules: even-^ count, no ^^^, no leading ^^")
    print("="*65)

    r_cond  = evaluate_csv(COND_CSV,  "generated_cond",  TRAIN_CSV,
                           "Conditioned no-CFG (80% mask, cluster N=133)")
    r_uncond = evaluate_csv(UNCOND_CSV, "generated_uncond", TRAIN_CSV,
                            "Unconditioned (80% mask, cluster N=133)")

    show_failure_examples(COND_CSV,  "generated_cond",  n=5)
    show_failure_examples(UNCOND_CSV, "generated_uncond", n=5)

    # Also check base-model unconditioned (local N=157)
    BASE_UNCOND = os.path.join(DATA_DIR, "gen_unconditioned_results.csv")
    if os.path.exists(BASE_UNCOND):
        r_base = evaluate_csv(BASE_UNCOND, "generated_unconditioned", TRAIN_CSV,
                              "Unconditioned base (15% mask, local N=157)")
        show_failure_examples(BASE_UNCOND, "generated_unconditioned", n=3)
