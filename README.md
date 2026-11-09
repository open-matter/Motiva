# Motiva: Polymer Language Model for Multi-Property Prediction and Inverse Design

**Motiva** is a RoBERTa-based transformer fine-tuned for sparse multi-task regression on polymer properties. It takes [GROVE](https://github.com/open-matter/GROVE) token sequences as input and predicts up to six properties simultaneously, handling ~60% missing labels per batch via property-specific loss masking.

**Paper:** "GROVE: A Universal Functional-Group Language for Data-Efficient Molecular Property Prediction" *(Nature Machine Intelligence, under review)*  
**Paper:** "Transformer-Driven Polymer Design via Synergy-Corrected Van Krevelen Group-Contribution Theory" *(Nature Materials, under review)*

---

## Key Results

| Task | Value |
|------|-------|
| Motiva test R² (5 props, avg) | **0.810** |
| polyVERSE bandgap fine-tuned R² | **0.755** |
| MoleculeNet BBBP AUC | **0.906** (matches MoLFormer) |
| PCMLM generation L2 validity | **100%** (conditioned) |
| GA inverse design (3-constraint pass) | **98/100** candidates |

---

## Installation

```bash
pip install torch transformers rdkit numpy pandas scikit-learn lightgbm shap
# also install GROVE:
pip install git+https://github.com/open-matter/GROVE.git
```

---

## Repository Structure

```
Motiva/
├── train.py                  # Motiva multi-task training (PolyInfo-1968 dataset)
├── model/
│   ├── encoder_roberta.py    # RoBERTa encoder wrapper
│   └── regressor_model.py    # Multi-task regression head
├── tokenizer/
│   ├── PolymerSumTokenization.py   # HuggingFace-compatible GROVE tokenizer
│   ├── tokenizer_generate.py
│   └── config/               # Vocabulary files (config.json, dict.txt, merged_vocab_rl.json, …)
├── pcmlm/
│   ├── gen_highmasked.py     # PCMLM: property-conditioned generation (high-mask fine-tuning)
│   ├── gen_cfg.py            # PCMLM with classifier-free guidance (CFG)
│   └── eval_validity.py      # GROVE L2 structural validity evaluation
├── analysis/
│   ├── run_attribution.py    # LightGBM + SHAP attribution pipeline
│   ├── synergy_analysis.py   # Non-additive FG-pair synergy scores
│   ├── van_krevelen.py       # Van Krevelen group-contribution baseline
│   ├── analysis_new_physics.py   # Baeyer strain, transfer matrix, decoupling
│   └── analysis_deep_physics.py  # Uncertainty principle, Boltzmann T_eff, atomic hierarchy
├── inverse_design/
│   ├── genetic_optimizer.py  # GA inverse design (GROVE mutation operators)
│   ├── run_inverse_design.py
│   ├── predictor_oracle.py   # Motiva as fitness oracle
│   └── pareto_explorer.py    # Multi-objective Pareto frontier
└── data/
    ├── polyinfo_1968/        # PolyInfo-derived training dataset (1968 polymers × 5 properties)
    │   ├── data_ALL.csv
    │   ├── data_split.csv
    │   ├── data_split_AUG.csv  # Augmented splits (SMILES randomization)
    │   └── data_test.csv
    └── results/              # Pre-computed results
        ├── fg_attribution_shap.csv       # SHAP values: 64 FGs × 6 properties
        ├── fg_synergy_empirical.csv      # 7488 FG-pair synergy scores
        ├── fg_synergy_top20.csv
        ├── all_candidates_ranked.csv     # 100 GA inverse design candidates
        ├── novelty_check.csv             # Tanimoto similarity to training set
        └── pcmlm_dft_results.csv         # DFT validation (B3LYP/6-31G*)
```

---

## Training Motiva

```bash
python train.py
# Trains on data/polyinfo_1968/data_split_AUG.csv
# Saves checkpoint to: checkpoints/train_final.pt
```

The training script implements:
- Multi-task head with per-property loss masking
- Uncertainty-weighted task balancing (Kendall & Gal 2018)
- SMILES randomization data augmentation

**Model checkpoint** is not included in this repository. To obtain the pre-trained weights, please contact the authors.

---

## PCMLM Generation

Property-conditioned polymer generation using high-mask MLM fine-tuning:

```bash
# Cluster (SLURM) usage:
python pcmlm/gen_highmasked.py \
    --checkpoint /path/to/pcmlm_best.pt \
    --target_props "000004000300000000000" \
    --n_samples 100
```

The numerical conditioning format encodes target property values as a 21-character string prefix. See Methods section of the paper for the encoding specification.

---

## Analysis Pipeline

```bash
# 1. SHAP attribution
python analysis/run_attribution.py

# 2. Synergy analysis
python analysis/synergy_analysis.py

# 3. Inverse design (GA)
python inverse_design/run_inverse_design.py
```

---

## Citation

```
@article{motiva2026nmi,
  title   = {GROVE: A Universal Functional-Group Language for Data-Efficient Molecular Property Prediction},
  author  = {Laplacian},
  journal = {Nature Machine Intelligence},
  year    = {2026},
  note    = {Under review}
}

@article{motiva2026nm,
  title   = {Transformer-Driven Polymer Design via Synergy-Corrected Van Krevelen Group-Contribution Theory},
  author  = {Laplacian},
  journal = {Nature Materials},
  year    = {2026},
  note    = {Under review}
}
```
