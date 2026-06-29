
import glob
from pathlib import Path
import itertools

import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

CSV_DIR_REPAINT = "./results/"
AURC_AUDC_Seed0  = "./results/aurc_seed0.csv"
AURC_AUDC_Seed42 = "./results/aurc_seed42.csv"
AURC_AUDC_Seed123 = "./results/aurc_seed123.csv"
Original_seed = "PLACEHOLDER_missing_reference.csv"  # TODO: locate original AURC reference CSV from prior experiment

LARGER_IS_BETTER = {
    'aurc': False,
    'audc': False,
}


def rank_tbl(tbl: pd.DataFrame | pd.Series, larger_is_better: bool):
    if isinstance(tbl, pd.Series):
        tbl = tbl.to_frame()
    return tbl.rank(ascending=not larger_is_better, axis=0)

def spearman_pairwise(vectors: dict[str, pd.Series | pd.DataFrame]):

    names = list(vectors)
    rho_mat = pd.DataFrame(index=names, columns=names, dtype=float)

    for a, b in itertools.combinations_with_replacement(names, 2):
        v1 = pd.Series(vectors[a]).astype(float)
        v2 = pd.Series(vectors[b]).astype(float)
        rho, _ = spearmanr(v1, v2)
        rho_mat.loc[a, b] = rho_mat.loc[b, a] = rho
    return rho_mat

def add_prefix(df: pd.DataFrame, prefix: str):
    new = df.copy()
    new.columns = [f"{prefix}_{c}" for c in new.columns]
    return new


# ---- LRC / AURC ...
methods = ["gradcam", "gbp", "smoothgrad", "grads", "ig", "sal"]


aurc_df_seed0 = pd.read_csv(AURC_AUDC_Seed0)
aurc_df_seed42 = pd.read_csv(AURC_AUDC_Seed42)
aurc_df_seed123 = pd.read_csv(AURC_AUDC_Seed123)
aurc_df_original = pd.read_csv(Original_seed)

def _mean_for_methods(df: pd.DataFrame, methods: list[str], prefer_masked: bool=False):
    vals = []
    for m in methods:
        # prefer explicit audc/aurc columns if present
        col = None
        # search for exact patterns first
        candidates = []
        if prefer_masked:
            candidates = [f"audc_{m}", f"aurc_masked_{m}", f"audc_masked_{m}"]
        else:
            candidates = [f"aurc_{m}", f"aurc_r_{m}"]
        for c in candidates:
            if c in df.columns:
                col = c
                break

        # fallback: find any column that contains the method name and keywords
        if col is None:
            for c in df.columns:
                if m in c:
                    if prefer_masked:
                        if 'mask' in c or 'audc' in c:
                            col = c
                            break
                    else:
                        if 'aurc' in c and 'mask' not in c:
                            col = c
                            break

        if col is None:
            # if still not found, append NaN so script can continue
            vals.append(float('nan'))
        else:
            vals.append(df[col].mean())

    s = pd.Series(vals, index=methods)
    return s

# Extract AURC scores for each seed (lower is better)
aurc_seed0 = _mean_for_methods(aurc_df_seed0, methods, prefer_masked=False)
aurc_seed0.name = 'seed0'
aurc_seed42 = _mean_for_methods(aurc_df_seed42, methods, prefer_masked=False)
aurc_seed42.name = 'seed42'
aurc_seed123 = _mean_for_methods(aurc_df_seed123, methods, prefer_masked=False)
aurc_seed123.name = 'seed123'
aurc_original = _mean_for_methods(aurc_df_original, methods, prefer_masked=False)
aurc_original.name = 'original'

# Extract AUDC scores for each seed (lower is better)
audc_seed0 = _mean_for_methods(aurc_df_seed0, methods, prefer_masked=True)
audc_seed0.name = 'seed0'
audc_seed42 = _mean_for_methods(aurc_df_seed42, methods, prefer_masked=True)
audc_seed42.name = 'seed42'
audc_seed123 = _mean_for_methods(aurc_df_seed123, methods, prefer_masked=True)
audc_seed123.name = 'seed123'
audc_original = _mean_for_methods(aurc_df_original, methods, prefer_masked=True)
audc_original.name = 'original'

print("\n=== AURC Scores ===")
aurc_all_seeds = pd.DataFrame([aurc_seed0, aurc_seed42, aurc_seed123, aurc_original]).T
print(aurc_all_seeds)

print("\n=== AUDC Scores ===")
audc_all_seeds = pd.DataFrame([audc_seed0, audc_seed42, audc_seed123, audc_original]).T
print(audc_all_seeds)

# Calculate Spearman correlation between seeds for AURC
print("\n=== Spearman Rank Correlation between Seeds (AURC) ===")
aurc_seeds_dict = {'seed0': aurc_seed0, 'seed42': aurc_seed42, 'seed123': aurc_seed123, 'original': aurc_original}
aurc_spearman_matrix = spearman_pairwise(aurc_seeds_dict)
print(aurc_spearman_matrix)

# Calculate Spearman correlation between seeds for AUDC
print("\n=== Spearman Rank Correlation between Seeds (AUDC) ===")
audc_seeds_dict = {'seed0': audc_seed0, 'seed42': audc_seed42, 'seed123': audc_seed123, 'original': audc_original}
audc_spearman_matrix = spearman_pairwise(audc_seeds_dict)
print(audc_spearman_matrix)

# Save results
out_path_aurc = Path("logs/spearman_seeds_aurc.csv")
out_path_aurc.parent.mkdir(parents=True, exist_ok=True)
aurc_spearman_matrix.to_csv(out_path_aurc)
print(f"\nAURC Spearman correlation saved to {out_path_aurc}")

out_path_audc = Path("logs/spearman_seeds_audc.csv")
audc_spearman_matrix.to_csv(out_path_audc)
print(f"AUDC Spearman correlation saved to {out_path_audc}")

# Visualize AURC correlation
plt.figure(figsize=(8, 6))
sns.heatmap(aurc_spearman_matrix, annot=True, cmap='Blues', fmt=".3f", linewidths=.5, vmin=0, vmax=1)
plt.title("Spearman Rank Correlation - AURC(lower) Scores Across Seeds")
plt.tight_layout()
plt.savefig("logs/spearman_aurc_seeds.png", dpi=150)
print("AURC visualization saved to logs/spearman_aurc_seeds.png")

# Visualize AUDC correlation
plt.figure(figsize=(8, 6))
sns.heatmap(audc_spearman_matrix, annot=True, cmap='Blues', fmt=".3f", linewidths=.5, vmin=0, vmax=1)
plt.title("Spearman Rank Correlation - AUDC Scores Across Seeds\n(Lower is Better)")
plt.tight_layout()
plt.savefig("logs/spearman_audc_seeds.png", dpi=150)
print("AUDC visualization saved to logs/spearman_audc_seeds.png")
plt.show()

# OLD CODE BELOW (keeping for reference)
aurc_mean = _mean_for_methods(aurc_df_seed0, methods, prefer_masked=False)
aurc_mean.name = 'aurc'
audc_mean = _mean_for_methods(aurc_df_seed0, methods, prefer_masked=True)
audc_mean.name = 'audc'

# # AUIC_R / AUIC_MASKED
# auic_df = pd.read_csv(AUIC_FILE)
# auic_r_mean = auic_df[[f"auic_r_{m}" for m in methods]].mean()
# auic_r_mean.index = methods
# auic_r_mean.name = "auic_r"

# OLD CODE - COMMENTED OUT
# auic_masked_mean = auic_df[[f"auic_masked_{m}" for m in methods]].mean()
# auic_masked_mean.index = methods
# auic_masked_mean.name = "auic_masked"

# res_global = pd.DataFrame([aurc_mean, audc_mean])[methods].T

# # compute ranks only for available columns in res_global
# ranks = {}
# for col in res_global.columns:
#     bigger = LARGER_IS_BETTER.get(col, False)
#     ranks[col] = rank_tbl(res_global[col], bigger)

# print("\nranks LRC/AURC/AUDC/AUIC (repaint):")
# print(ranks)
# print(ranks.keys())

# # Consensus 
# consensus = {}
# metrics = list(ranks.keys())        
# for m in metrics:
#     result = {}
#     for n in metrics:
#         rho, pvalue = spearmanr(ranks[m].values, ranks[n].values)
#         res = f"{round(rho,2)}±{round(pvalue,2)}"
#         result[n] = res
#     consensus[m] = pd.Series(result)
# consensus = pd.DataFrame(consensus).T
# print("\nSpearman consensus:")
# print(consensus)

# # save consensus to CSV (ensure directory exists)
# out_path = Path("logs/seed0/spearman_consensus_seed0.csv")
# out_path.parent.mkdir(parents=True, exist_ok=True)
# consensus.to_csv(out_path)

# # visualisation of consensus using matplotlib cmap=Blues
# # consensus currently contains strings like 'rho±pvalue' — extract numeric rho for plotting
# def _extract_rho(val):
#     try:
#         if isinstance(val, str) and '±' in val:
#             return float(val.split('±')[0])
#         return float(val)
#     except Exception:
#         return float('nan')

# numeric_consensus = consensus.copy()
# numeric_consensus = numeric_consensus.applymap(_extract_rho)

# plt.figure(figsize=(10, 8))
# sns.heatmap(numeric_consensus, annot=True, cmap='Blues', fmt=".2f", linewidths=.5, cbar_kws={"shrink": .8})
# plt.title("Spearman Rank Correlation Consensus")
# plt.xticks(rotation=45)
# plt.yticks(rotation=0)
# plt.tight_layout()
# plt.show()  
