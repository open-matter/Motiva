from __future__ import annotations
"""
Predictor Oracle
=================
Wraps the trained multi-task regressor as a callable oracle for inverse design.
Accepts a list of SMILES and returns:
  - predicted property values (N Ã— P)
  - uncertainty estimates (N Ã— P)  via MC-Dropout

Also provides a fitness scorer for single-objective and multi-objective optimization.
"""
import sys
import numpy as np
import torch
from pathlib import Path

POLY_DESIGN = Path(__file__).parent.parent
sys.path.insert(0, str(POLY_DESIGN))

from data.loaders      import load_config
from poly_encodings.sum_encoding import smiles_to_sum_features, batch_smiles_to_sum

cfg       = load_config()
PROP_NAMES = cfg["properties"]["names"]


# â”€â”€ Oracle class â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class PredictorOracle:
    """
    Wraps the SUM-RoBERTa regressor for fast batch inference.
    Can use either:
      - the pre-trained SUM regressor (requires checkpoint)
      - a simpler MLP trained on Morgan FP (fallback)
    """

    def __init__(self, model=None, tokenizer=None,
                 fallback_model=None, fallback_scaler=None,
                 prop_names: list[str] | None = None,
                 device: str | None = None,
                 output_scalers=None):
        self.prop_names     = prop_names or PROP_NAMES
        self.device         = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model          = model
        self.tokenizer      = tokenizer
        self.fallback_model = fallback_model
        self.fallback_scaler= fallback_scaler
        self.output_scalers = output_scalers  # dict {prop_name: sklearn StandardScaler}

    @classmethod
    def from_checkpoint(cls, cfg=None):
        """Build oracle from E2E fine-tuned SUMRegressorModel (preferred)
        or fall back to Morgan MLP if not available."""
        if cfg is None:
            cfg = load_config()
        prop_names = cfg["properties"]["names"]

        # Load output scalers (inverse-transform normalized predictions → physical units)
        output_scalers = None
        scalers_path = POLY_DESIGN / "1_encoding_comparison" / "results" / "scalers.pkl"
        if scalers_path.exists():
            import pickle
            with open(scalers_path, "rb") as f:
                output_scalers = pickle.load(f)
            print(f"[oracle] Loaded output scalers for {list(output_scalers.keys())}")

        # Try E2E fine-tuned model first
        e2e_ckpt = POLY_DESIGN / "1_encoding_comparison" / "results" / "e2e_sum_model.pt"
        if e2e_ckpt.exists() and e2e_ckpt.stat().st_size > 1024:
            try:
                from models.sum_regressor import SUMRegressorModel, build_tokenizer
                model = SUMRegressorModel(output_dim=len(prop_names))
                state = torch.load(str(e2e_ckpt), map_location="cpu")
                model.load_state_dict(state)
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = model.to(device)
                model.eval()
                tokenizer = build_tokenizer()
                print(f"[oracle] Loaded E2E fine-tuned model from {e2e_ckpt}")
                return cls(model=model, tokenizer=tokenizer, prop_names=prop_names,
                           output_scalers=output_scalers)
            except Exception as e:
                print(f"[oracle] E2E model load failed: {e}")

        # Fall back to pre-trained encoder only (no regression head — limited accuracy)
        try:
            from poly_encodings.sum_encoding import _load_model
            tokenizer, encoder = _load_model()
            if encoder is not None:
                print("[oracle] Warning: using pre-trained encoder (no regression head)")
            return cls(model=encoder, tokenizer=tokenizer, prop_names=prop_names)
        except Exception as e:
            print(f"[oracle] Could not load any model: {e}")
            return cls(prop_names=prop_names)

    @classmethod
    def from_mlp(cls, mlp_model, scaler, cfg=None):
        """Build oracle from a lightweight MLP trained on Morgan FP."""
        if cfg is None:
            cfg = load_config()
        return cls(fallback_model=mlp_model, fallback_scaler=scaler,
                   prop_names=cfg["properties"]["names"])

    def predict(self, smiles_list: list[str],
                batch_size: int = 32) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (mean, std) arrays of shape (N, P).
        Uses model if available, otherwise fallback.
        """
        if self.model is not None and self.tokenizer is not None:
            return self._predict_roberta(smiles_list, batch_size)
        elif self.fallback_model is not None:
            return self._predict_mlp(smiles_list)
        else:
            n = len(smiles_list)
            p = len(self.prop_names)
            return np.zeros((n, p)), np.ones((n, p))

    @torch.no_grad()
    def _predict_roberta(self, smiles_list, batch_size):
        self.model.eval()
        sum_strings = batch_smiles_to_sum(smiles_list, show_progress=False)
        all_preds   = []
        # Use SUM string if available, fall back to raw SMILES
        valid = [s if s else smi for s, smi in zip(sum_strings, smiles_list)]

        for start in range(0, len(valid), batch_size):
            batch = valid[start: start + batch_size]
            enc   = self.tokenizer(batch, padding="max_length", max_length=128,
                                   truncation=True, return_tensors="pt")
            ids   = enc["input_ids"].to(self.device)
            amask = enc["attention_mask"].to(self.device)
            out   = self.model(ids, amask)
            # SUMRegressorModel → (B, P); EncoderRoberta → (B, L, H) mean pool
            if out.dim() == 3:
                out = out.mean(dim=1)
            all_preds.append(out.cpu().numpy())

        preds = np.concatenate(all_preds, axis=0)
        # Inverse-transform normalized predictions → physical units
        if self.output_scalers:
            for i, prop in enumerate(self.prop_names):
                if prop in self.output_scalers:
                    sc = self.output_scalers[prop]
                    preds[:, i] = preds[:, i] * sc.scale_[0] + sc.mean_[0]
        std   = np.zeros_like(preds)
        return preds, std

    def _predict_mlp(self, smiles_list):
        from poly_encodings.fingerprint_encoding import smiles_to_morgan
        X = smiles_to_morgan(smiles_list)
        if self.fallback_scaler:
            X = self.fallback_scaler.transform(X)
        X_t = torch.tensor(X, dtype=torch.float32)
        if hasattr(self.fallback_model, "predict_with_uncertainty"):
            mean, std = self.fallback_model.predict_with_uncertainty(X_t, n_samples=20)
        else:
            with torch.no_grad():
                mean = self.fallback_model(X_t).numpy()
            std = np.zeros_like(mean)
        return mean.astype(np.float32), std.astype(np.float32)

    def fitness(self, smiles_list: list[str],
                targets: dict[str, tuple[float, float]],
                uncertainty_penalty: float = 0.1) -> np.ndarray:
        """
        Multi-property fitness score âˆ?[0, 1] per polymer.
        targets: {property_name: (min_value, max_value)}
        Returns array of shape (N,).
        """
        mean, std = self.predict(smiles_list)
        n = len(smiles_list)
        scores = np.ones(n, dtype=np.float32)

        for prop, (lo, hi) in targets.items():
            if prop not in self.prop_names:
                continue
            idx  = self.prop_names.index(prop)
            vals = mean[:, idx]
            unc  = std[:, idx]
            # Gaussian-shaped reward: 1 at center, 0 at edges
            center   = (lo + hi) / 2
            width    = (hi - lo) / 2
            in_range = np.clip(1 - np.abs(vals - center) / (width + 1e-8), 0, 1)
            penalty  = uncertainty_penalty * unc
            scores  *= np.clip(in_range - penalty, 0, 1)

        return scores

    def pareto_dominance(self, smiles_list: list[str],
                          objectives: list[tuple[str, str]]) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute Pareto front for multi-objective optimization.
        objectives: list of (property_name, "max" or "min")
        Returns (pareto_mask, objective_values).
        """
        mean, std = self.predict(smiles_list)
        n_obj = len(objectives)
        obj_vals = np.zeros((len(smiles_list), n_obj))

        for k, (prop, direction) in enumerate(objectives):
            if prop not in self.prop_names:
                continue
            idx = self.prop_names.index(prop)
            obj_vals[:, k] = mean[:, idx] * (1 if direction == "max" else -1)

        pareto_mask = _is_pareto_efficient(obj_vals)
        return pareto_mask, obj_vals * np.array([1 if d == "max" else -1
                                                  for _, d in objectives])


def _is_pareto_efficient(costs: np.ndarray) -> np.ndarray:
    """Returns boolean mask of Pareto-efficient solutions (maximization)."""
    n = len(costs)
    is_efficient = np.ones(n, dtype=bool)
    for i in range(n):
        if is_efficient[i]:
            dominated = np.all(costs >= costs[i], axis=1) & \
                        np.any(costs > costs[i], axis=1)
            dominated[i] = False
            is_efficient[dominated] = False
    return is_efficient


if __name__ == "__main__":
    oracle = PredictorOracle.from_checkpoint()
    test   = ["[*]c1ccc(cc1)C(=O)O", "[*]CCCCCC[*]"]
    m, s   = oracle.predict(test)
    print(f"Predictions shape: {m.shape}")
    scores = oracle.fitness(test, targets={"Tg": (400, 600), "Egb": (1.5, 3.0)})
    print(f"Fitness scores: {scores}")
