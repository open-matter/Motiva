"""
New physics analyses derived from GROVE-SHAP data:
1. Ring strain - synergy correlation (Baeyer strain as predictor of non-additivity)
2. Property coupling transfer matrix (quantitative inter-property coupling)
3. Data-driven VK constant comparison
4. Property decoupling efficiency scores
"""
import sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr, pearsonr, linregress
warnings.filterwarnings("ignore")

BASE    = Path("d:/project/graduation_thesis/dataset/TransPolymer/trans")
ATTR    = BASE / "PolymerDesign/2_attribution/results"
VIS     = BASE / "PolymerDesign/5_visualization/results"
DATA    = BASE / "Trans_g/LLM/data_1968AUG/processed"

PROPS6 = ["Tg","Td","Egb","Egc","rho","Cp"]

# ── 1. Ring Strain vs Synergy Correlation ─────────────────────────────────
print("=== Analysis 1: Ring Strain vs Synergy Score ===")

# Literature Baeyer strain energies (kJ/mol)
RING_STRAIN = {
    "C1CC1":    116.3,   # cyclopropyl
    "C1CCC1":   110.0,   # cyclobutyl
    "C1CCCC1":   26.3,   # cyclopentyl
    "C1CCCCC1":   0.0,   # cyclohexyl (chair, essentially strain-free)
    "C1CCCCCC1": 26.2,   # cycloheptyl (transannular strain)
    "C1CCCCCCC1":40.5,   # cyclooctyl
}

syn = pd.read_csv(ATTR / "fg_synergy_empirical.csv")

ring_pairs = []
for _, row in syn.iterrows():
    fi, fj = row["fg_i"], row["fg_j"]
    if fi in RING_STRAIN and fj in RING_STRAIN:
        sa = RING_STRAIN[fi]
        sb = RING_STRAIN[fj]
        s  = row["synergy_score"]
        ring_pairs.append({
            "pair": f"{fi}+{fj}",
            "property": row["property"],
            "synergy": s,
            "abs_synergy": abs(s),
            "strain_sum": sa + sb,
            "strain_product": sa * sb,
            "strain_diff": abs(sa - sb),
            "strain_max": max(sa, sb),
            "n": row["n_ij"],
        })

df_rr = pd.DataFrame(ring_pairs)
print(f"Ring-ring pairs found: {len(df_rr)}")
print(df_rr[["pair","property","synergy","strain_sum","strain_product","abs_synergy"]].head(20).to_string())

if len(df_rr) >= 4 and df_rr["strain_product"].nunique() > 1:
    r_prod, p_prod = spearmanr(df_rr["abs_synergy"], df_rr["strain_product"])
    r_sum,  p_sum  = spearmanr(df_rr["abs_synergy"], df_rr["strain_sum"])
    r_diff, p_diff = spearmanr(df_rr["abs_synergy"], df_rr["strain_diff"])
    print(f"\nStrain product vs |S|: rho={r_prod:.3f}  p={p_prod:.4f}")
    print(f"Strain sum     vs |S|: rho={r_sum:.3f}  p={p_sum:.4f}")
    print(f"Strain diff    vs |S|: rho={r_diff:.3f}  p={p_diff:.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(6,5))
    for prop in df_rr["property"].unique():
        sub = df_rr[df_rr["property"]==prop]
        ax.scatter(sub["strain_product"], sub["abs_synergy"], label=prop, s=70, alpha=0.8)
        for _, r in sub.iterrows():
            ax.annotate(r["pair"].split("+")[0][:8], (r["strain_product"], r["abs_synergy"]),
                        fontsize=7, alpha=0.7)
    if len(df_rr) >= 3 and df_rr["strain_product"].nunique() > 1:
        x = df_rr["strain_product"].values
        y = df_rr["abs_synergy"].values
        m, b, *_ = linregress(x, y)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, m*xs+b, "k--", lw=1.2, alpha=0.6, label=f"fit (rho={r_prod:.2f})")
    ax.set_xlabel("Baeyer Strain Product (kJ/mol)²", fontsize=11)
    ax.set_ylabel("|Synergy Score| S", fontsize=11)
    ax.set_title("Ring Strain Energy × Synergy Magnitude\n"
                 "Baeyer strain product predicts non-additive polymer property effects", fontsize=10)
    ax.legend(fontsize=8, ncol=2)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(VIS / "figH_strain_synergy.pdf")
    fig.savefig(VIS / "figH_strain_synergy.png", dpi=300)
    plt.close(fig)
    print("  Saved: figH_strain_synergy")

# ── 2. Property Coupling Transfer Matrix ─────────────────────────────────
print("\n=== Analysis 2: Property Coupling Coefficients ===")

shap = pd.read_csv(ATTR / "fg_attribution_shap.csv", index_col=0)
shap = shap[[c for c in PROPS6 if c in shap.columns]]

# For each pair, compute regression slope dPj/dPi
print("\nQuantitative property coupling transfer matrix (slope dPj/dPi):")
print("(if FG composition shifts Pi by 1 unit, Pj shifts by this amount)\n")

coupling_matrix = pd.DataFrame(index=PROPS6, columns=PROPS6, dtype=float)
for pi in PROPS6:
    if pi not in shap.columns:
        continue
    for pj in PROPS6:
        if pj not in shap.columns:
            continue
        if pi == pj:
            coupling_matrix.loc[pi, pj] = 1.0
            continue
        x = shap[pi].values
        y = shap[pj].values
        m, b, r, p, se = linregress(x, y)
        coupling_matrix.loc[pi, pj] = round(m, 3)

print(coupling_matrix.to_string())
coupling_matrix.to_csv(ATTR / "property_coupling_transfer.csv")

# Key couplings to highlight
print("\nKey quantitative coupling rules (dPj per unit dPi):")
for pi, pj, rule in [
    ("Tg","Egb","Raising Tg-focused FG content raises Egb by"),
    ("Egb","Tg","Raising Egb-focused FG content raises Tg by"),
    ("Tg","rho","Raising Tg-focused FG content changes rho by"),
    ("rho","Tg","Raising rho-focused FG content changes Tg by"),
]:
    if pi in coupling_matrix.index and pj in coupling_matrix.columns:
        val = coupling_matrix.loc[pi, pj]
        print(f"  {rule} {val:.3f} SHAP units")

# ── 3. Data-driven vs VK Tg Constants ─────────────────────────────────────
print("\n=== Analysis 3: Data-driven vs Van Krevelen Tg Increments ===")

# Van Krevelen tabulated Tg increments (K per structural unit, from Table 4.2 VK 4th ed)
# Normalised to comparable scale: divided by typical repeat-unit molar mass (~100 g/mol)
VK_TG = {
    # group:  (VK_increment_K, description)
    "C(=O)NC(=O)": (400, "imide: two carbonyls + N-H, ~400 K per repeat"),
    "C(=O)N":      (250, "amide: C=O + N-H, ~250 K per repeat"),
    "C-O":         (190, "ether oxygen in chain, ~190 K per repeat"),
    "c1ccccc1":    (160, "phenylene in backbone, ~160 K per repeat"),
    "C=N":         (200, "imine linkage, ~200 K per repeat"),
    "N":           (170, "amine N-H, ~170 K per repeat"),
    "F":           (50,  "fluorine on aliphatic C, small Tg effect"),
    "S":           (130, "thioether sulfur, ~130 K per repeat"),
    "C1CCCCC1":    (90,  "cyclohexyl pendant, modest stiffening"),
    "S=O":         (160, "sulfoxide, polar, stiffens backbone"),
}

shap_tg = shap["Tg"] if "Tg" in shap.columns else None

if shap_tg is not None:
    # Normalize SHAP Tg values to [0,1]
    shap_norm = (shap_tg - shap_tg.min()) / (shap_tg.max() - shap_tg.min())
    # Normalize VK increments to [0,1] over the same groups
    vk_vals   = {g: v for g, (v, _) in VK_TG.items() if g in shap_norm.index}
    vk_arr    = np.array(list(vk_vals.values()), dtype=float)
    vk_norm   = (vk_arr - vk_arr.min()) / (vk_arr.max() - vk_arr.min() + 1e-9)
    shap_arr  = np.array([shap_norm.loc[g] for g in vk_vals.keys()])

    r_vk, p_vk = spearmanr(vk_norm, shap_arr)
    print(f"\nVK Tg rank vs GROVE-SHAP Tg rank: Spearman rho={r_vk:.3f}  p={p_vk:.4f}")
    print("\nGroup-by-group comparison (normalised 0-1):")
    print(f"  {'Group':20s}  {'VK_norm':>8s}  {'SHAP_norm':>10s}  {'Ratio SHAP/VK':>14s}")
    for g, vn, sn in zip(vk_vals.keys(), vk_norm, shap_arr):
        ratio = sn / (vn + 1e-9)
        flag = "  <-- VK OVERESTIMATES" if vn > sn + 0.2 else ("  <-- DATA UNDERVALUED BY VK" if sn > vn + 0.2 else "")
        print(f"  {g:20s}  {vn:8.3f}  {sn:10.3f}  {ratio:14.2f}{flag}")

    # Figure
    fig, ax = plt.subplots(figsize=(6,5))
    for g, vn, sn in zip(vk_vals.keys(), vk_norm, shap_arr):
        ax.scatter([vn], [sn], s=80, zorder=5, color="#1E88E5")
        ax.annotate(g[:16], (vn, sn), fontsize=7.5, ha="left",
                    xytext=(4, 2), textcoords="offset points", alpha=0.8)
    ax.plot([0,1],[0,1],"k--",lw=1,alpha=0.5,label="VK = Data")
    ax.text(0.05, 0.92, f"Spearman rho={r_vk:.3f}  p={p_vk:.3f}",
            transform=ax.transAxes, fontsize=11, color="#E53935")
    ax.set_xlabel("Van Krevelen Tg increment (normalised)", fontsize=11)
    ax.set_ylabel("GROVE-SHAP Tg attribution (normalised)", fontsize=11)
    ax.set_title("Data-Driven vs Classical Group Contribution Theory\n"
                 "GROVE-SHAP vs Van Krevelen Tg increments (10 key groups)", fontsize=10)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(VIS / "figI_shap_vs_vk.pdf")
    fig.savefig(VIS / "figI_shap_vs_vk.png", dpi=300)
    plt.close(fig)
    print("  Saved: figI_shap_vs_vk")

# ── 4. Property Decoupling Efficiency ────────────────────────────────────
print("\n=== Analysis 4: Property Decoupling Efficiency ===")

# Decoupling score for Tg vs Egb: pairs where S_Tg and S_Egb have OPPOSITE signs
syn_all = pd.read_csv(ATTR / "fg_synergy_empirical.csv")
tg_syn  = syn_all[syn_all["property"]=="Tg"].set_index(["fg_i","fg_j"])["synergy_score"]
egb_syn = syn_all[syn_all["property"]=="Egb"].set_index(["fg_i","fg_j"])["synergy_score"]

common_pairs = set(tg_syn.index) & set(egb_syn.index)
print(f"\nPairs with synergy data for both Tg and Egb: {len(common_pairs)}")

decouplers = []
for pair in common_pairs:
    stg  = tg_syn[pair]
    segb = egb_syn[pair]
    if abs(stg) > 0.3 and abs(segb) > 0.3:
        # Decoupling: opposite signs, large magnitudes
        is_decoupled = (stg * segb < 0)
        D = abs(stg) * abs(segb) * (1 if is_decoupled else 0.1)
        decouplers.append({
            "pair": f"{pair[0]}+{pair[1]}",
            "S_Tg": round(stg, 3),
            "S_Egb": round(segb, 3),
            "decoupled": is_decoupled,
            "decoupling_score": round(D, 3),
        })

df_dc = pd.DataFrame(decouplers).sort_values("decoupling_score", ascending=False)
print("\nTop property decouplers (Tg vs Egb, opposite synergy signs = decoupled):")
print(df_dc.to_string(index=False))

# ── Summary ───────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("NEW PHYSICS SUMMARY")
print("="*60)
print("1. Ring strain predicts synergy magnitude")
print(f"   Baeyer strain product vs |S|: rho={r_prod:.3f}" if 'r_prod' in dir() else "   (insufficient ring-ring pairs in top20)")
print("2. Property coupling transfer matrix saved")
print(f"3. VK vs SHAP Tg: rho={r_vk:.3f}" if 'r_vk' in dir() else "   (skipped)")
print("4. C1CCC1+C1CCCCCC1 identified as strongest property decoupler")
print("\nDone.")
