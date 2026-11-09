from __future__ import annotations
"""
Genetic Algorithm for Polymer Inverse Design
=============================================
Starts from a pool of known polymers and evolves them toward target properties
using SMILES-level mutations (add/remove/replace functional groups).

Chromosome: SMILES string (polymer repeat unit)
Fitness:    multi-property satisfaction score from PredictorOracle

Operators:
  - Mutation 1:  Add a functional group substituent at a random aromatic position
  - Mutation 2:  Remove a substituent branch
  - Mutation 3:  Replace an atom type (C→N, C→O, etc.)
  - Mutation 4:  Change ring size or aromaticity
  - Crossover:   Fragment swap between two SMILES

Outputs:
  results/ga_evolution.csv     -- fitness history per generation
  results/ga_top_candidates.csv -- top-k final candidates
"""
import sys
import random
import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy

POLY_DESIGN = Path(__file__).parent.parent
sys.path.insert(0, str(POLY_DESIGN))
sys.path.insert(0, str(Path(__file__).parent))

from data.loaders       import load_config, load_polyone
from synthesizability   import is_valid_smiles, is_valid_polymer, get_sa_score

RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdchem, rdmolops
    from rdkit.Chem.rdchem import RWMol
    RDKIT_OK = True
except ImportError:
    RDKIT_OK = False


# ── SMILES mutation operations ───────────────────────────────────────────────────

ATOM_TYPES   = ["C", "N", "O", "S", "F", "Cl"]
SUBSTITUENTS = [
    "C", "N", "O", "F", "Cl", "Br",
    "C(=O)", "C(=O)O", "C(=O)N", "C#N",
    "OC(=O)", "c1ccccc1", "C(F)(F)F",
]

def _mol_clean(smi: str):
    """Return RDKit mol, treating [*] as attachment points."""
    s = smi.replace("[*]", "[At]")
    return Chem.MolFromSmiles(s)

def _mol_to_smi(mol) -> str:
    smi = Chem.MolToSmiles(mol)
    return smi.replace("[At]", "[*]")


def mutate_atom_type(smiles: str) -> str | None:
    """Replace a random non-[*] atom with a different type."""
    if not RDKIT_OK:
        return None
    mol = _mol_clean(smiles)
    if mol is None:
        return None
    rw = RWMol(mol)
    candidates = [a.GetIdx() for a in rw.GetAtoms()
                  if a.GetSymbol() not in ("At", "H") and not a.IsInRing()]
    if not candidates:
        return None
    idx    = random.choice(candidates)
    old    = rw.GetAtomWithIdx(idx).GetSymbol()
    new_at = random.choice([a for a in ATOM_TYPES if a != old])
    try:
        rw.GetAtomWithIdx(idx).SetAtomicNum(
            Chem.Atom(new_at).GetAtomicNum()
        )
        Chem.SanitizeMol(rw)
        return _mol_to_smi(rw)
    except Exception:
        return None


def mutate_add_substituent(smiles: str) -> str | None:
    """Add a substituent at a random aromatic carbon."""
    if not RDKIT_OK:
        return None
    mol = _mol_clean(smiles)
    if mol is None:
        return None
    candidates = [a.GetIdx() for a in mol.GetAtoms()
                  if a.GetIsAromatic() and a.GetTotalNumHs() > 0]
    if not candidates:
        return None
    idx   = random.choice(candidates)
    subst = random.choice(SUBSTITUENTS)
    try:
        new_smi = _mol_to_smi(mol)
        # Simple string manipulation: add substituent at a position
        # Real implementation would use RDKit EditableMol
        rw    = RWMol(mol)
        subst_mol = Chem.MolFromSmiles(subst)
        if subst_mol is None:
            return None
        combo = Chem.CombineMols(rw, subst_mol)
        edit  = RWMol(combo)
        n_orig = mol.GetNumAtoms()
        edit.AddBond(idx, n_orig, Chem.rdchem.BondType.SINGLE)
        Chem.SanitizeMol(edit)
        return _mol_to_smi(edit)
    except Exception:
        return None


def mutate_remove_branch(smiles: str) -> str | None:
    """Remove a terminal branch from SMILES."""
    if not RDKIT_OK:
        return None
    mol = _mol_clean(smiles)
    if mol is None:
        return None
    # Find leaf atoms (degree 1, not [*])
    leaves = [a.GetIdx() for a in mol.GetAtoms()
              if a.GetDegree() == 1 and a.GetSymbol() != "At"]
    if not leaves:
        return None
    idx = random.choice(leaves)
    rw = RWMol(mol)
    try:
        rw.RemoveAtom(idx)
        Chem.SanitizeMol(rw)
        result = _mol_to_smi(rw)
        return result if is_valid_polymer(result) else None
    except Exception:
        return None


def crossover_smiles(smi_a: str, smi_b: str) -> str | None:
    """Simple crossover: swap SMILES halves at a random '[' split point."""
    parts_a = smi_a.split("(")
    parts_b = smi_b.split("(")
    if len(parts_a) < 2 or len(parts_b) < 2:
        return None
    try:
        cut_a = random.randint(1, len(parts_a) - 1)
        cut_b = random.randint(1, len(parts_b) - 1)
        child = "(".join(parts_a[:cut_a] + parts_b[cut_b:])
        child = child.replace("[*]", "", 1) + "[*]"
        if is_valid_smiles(child) and is_valid_polymer(child):
            return child
    except Exception:
        pass
    return None


MUTATIONS = [mutate_atom_type, mutate_add_substituent, mutate_remove_branch]


def apply_mutation(smiles: str) -> str | None:
    """Apply a random mutation operator."""
    fn = random.choice(MUTATIONS)
    return fn(smiles)


# ── Genetic Algorithm ─────────────────────────────────────────────────────────────

class GeneticOptimizer:

    def __init__(self, oracle,
                 population_size: int = 100,
                 n_generations: int = 50,
                 mutation_rate: float = 0.15,
                 crossover_rate: float = 0.70,
                 elite_ratio: float = 0.10,
                 seed: int = 42):
        self.oracle         = oracle
        self.pop_size       = population_size
        self.n_gen          = n_generations
        self.mutation_rate  = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_ratio    = elite_ratio
        random.seed(seed)
        np.random.seed(seed)

    def run(self, seed_smiles: list[str],
            targets: dict[str, tuple[float, float]],
            verbose: bool = True) -> pd.DataFrame:
        """
        Main GA loop.
        seed_smiles: initial population (will be padded/truncated to pop_size)
        targets: {property: (min, max)}
        Returns DataFrame of all evaluated candidates with fitness scores.
        """
        # Initialize population
        population = list(seed_smiles[:self.pop_size])
        while len(population) < self.pop_size:
            population.append(random.choice(seed_smiles))

        history     = []
        all_visited = set(population)
        archive     = []  # (smiles, fitness)

        for gen in range(self.n_gen):
            # Evaluate fitness
            fitness = self.oracle.fitness(population, targets)

            # Archive top candidates
            for smi, fit in zip(population, fitness):
                archive.append({"smiles": smi, "fitness": float(fit),
                                 "generation": gen})

            best_fit = float(fitness.max())
            avg_fit  = float(fitness.mean())
            history.append({"generation": gen, "best": best_fit, "mean": avg_fit})

            if verbose and gen % 10 == 0:
                print(f"  Gen {gen:3d}  best={best_fit:.4f}  mean={avg_fit:.4f}")

            # Elite selection
            n_elite  = max(1, int(self.pop_size * self.elite_ratio))
            elite_idx = np.argsort(fitness)[-n_elite:]
            elites   = [population[i] for i in elite_idx]

            # Tournament selection
            new_pop = list(elites)
            while len(new_pop) < self.pop_size:
                # Tournament
                t_idx = np.random.choice(len(population), size=4, replace=False)
                t_fit = fitness[t_idx]
                p1    = population[t_idx[np.argmax(t_fit[:2])]]
                p2    = population[t_idx[np.argmax(t_fit[2:]) + 2]]

                # Crossover
                child = None
                if random.random() < self.crossover_rate:
                    child = crossover_smiles(p1, p2)

                if child is None:
                    child = p1

                # Mutation
                if random.random() < self.mutation_rate:
                    mutated = apply_mutation(child)
                    if mutated and mutated not in all_visited:
                        child = mutated
                        all_visited.add(child)

                if is_valid_smiles(child) and is_valid_polymer(child):
                    new_pop.append(child)
                else:
                    new_pop.append(random.choice(elites))

            population = new_pop[:self.pop_size]

        # Final evaluation
        final_fitness = self.oracle.fitness(population, targets)
        for smi, fit in zip(population, final_fitness):
            archive.append({"smiles": smi, "fitness": float(fit),
                             "generation": self.n_gen})

        results_df = pd.DataFrame(archive)
        results_df = results_df.drop_duplicates("smiles")
        results_df = results_df.sort_values("fitness", ascending=False)

        # Save history
        pd.DataFrame(history).to_csv(RESULTS / "ga_evolution.csv", index=False)
        results_df.head(100).to_csv(RESULTS / "ga_top_candidates.csv", index=False)

        if verbose:
            print(f"\n  GA complete. Top 5 candidates:")
            print(results_df[["smiles", "fitness", "generation"]].head(5).to_string(index=False))

        return results_df


if __name__ == "__main__":
    from predictor_oracle import PredictorOracle
    oracle = PredictorOracle.from_checkpoint()
    cfg    = load_config()
    df     = load_polyone(cfg)
    seeds  = df["smiles"].sample(50, random_state=42).tolist()
    ga     = GeneticOptimizer(oracle, population_size=50, n_generations=20)
    results = ga.run(seeds, targets={"Tg": (450, 600), "Egb": (2.0, 3.5)})
    print(results.head(10))
