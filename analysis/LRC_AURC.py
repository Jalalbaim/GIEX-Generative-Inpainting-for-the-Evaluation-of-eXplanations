"""
Ce code va récupérer les résultats de LRC, AURC, AUDC, 
Rank les resultats 
et calculer les metrics de Spearman Rank
@author: J.BAIM
"""

import pandas as pd
from scipy.stats import spearmanr

methods = ["gradcam", "gbp", "smoothgrad", "grads", "ig", "sal", "lrp"]

# ---- LRC ----
lrc_map = {"grad-cam": "gradcam",
           "integrated gradients": "ig",
           "saliency": "sal"}
lrc_mean = (pd.read_csv("./logs/lrc_results.csv")
              .rename(columns=lrc_map)[methods]
              .mean())
lrc_mean.name = "lrc"

# ---- AURC / AUDC ----
aurc_df = pd.read_csv("./results/aurc_seed42_lrp_included.csv")

# Helper to find a matching column for a method given possible prefixes
def _find_col_for_method(df, method, prefixes):
    for p in prefixes:
        col = f"{p}_{method}"
        if col in df.columns:
            return col
    return None

# For this dataset the AURC values live under 'aurc_r_<method>'
aurc_prefixes = ["aurc_r", "aurc"]
audc_prefixes = ["audc", "aurc_masked", "aurc_masked"]

aurc_values = []
aurc_used_cols = {}
for m in methods:
    col = _find_col_for_method(aurc_df, m, aurc_prefixes)
    if col:
        aurc_values.append(aurc_df[col].mean())
        aurc_used_cols[m] = col
    else:
        aurc_values.append(float("nan"))
        aurc_used_cols[m] = None
aurc_mean = pd.Series(aurc_values, index=methods)
aurc_mean.name = "aurc"

audc_values = []
audc_used_cols = {}
for m in methods:
    col = _find_col_for_method(aurc_df, m, audc_prefixes)
    if col:
        audc_values.append(aurc_df[col].mean())
        audc_used_cols[m] = col
    else:
        audc_values.append(float("nan"))
        audc_used_cols[m] = None
audc_mean = pd.Series(audc_values, index=methods)
audc_mean.name = "audc"

print("Detected AURC columns mapping:")
for m in methods:
    print(f"  {m}: aurc->{aurc_used_cols.get(m)}, audc->{audc_used_cols.get(m)}")

# ---- AUIC_R / AUIC_MASKED ----
# auic_df = pd.read_csv("RePaint/logs/metrics300/auic.csv")
# auic_r_mean = auic_df[[f"auic_r_{m}" for m in methods]].mean()
# auic_r_mean.index = methods
# auic_r_mean.name = "auic_r"

# auic_masked_mean = auic_df[[f"auic_masked_{m}" for m in methods]].mean()
# auic_masked_mean.index = methods
# auic_masked_mean.name = "auic_masked"

res = pd.DataFrame([lrc_mean, aurc_mean, audc_mean])[methods]
# res = pd.DataFrame([lrc_mean, aurc_mean, audc_mean])[methods]
res = res.transpose()
print(res)


# rank the results
def rank_tbl(tbl, larger_is_better=True):
    return tbl.rank(ascending=not larger_is_better, axis=0)

rank_lrc = rank_tbl(res["lrc"], larger_is_better=False)
rank_aurc = rank_tbl(res["aurc"], larger_is_better=False)
rank_audc = rank_tbl(res["audc"], larger_is_better=False)
# rank_auic_r = rank_tbl(res["auic_r"], larger_is_better=True)
# rank_auic_masked = rank_tbl(res["auic_masked"], larger_is_better=True)

print("\nRanked Results:")
print("LRC:\n", rank_lrc)
print("AURC:\n", rank_aurc)
print("AUDC:\n", rank_audc)
# print("AUIC Repainted:\n", rank_auic_r)
# print("AUIC Masked:\n", rank_auic_masked)

# spearman rank correlation
spearman_lrc = spearmanr(rank_lrc, rank_aurc)[0]
spearman_audc = spearmanr(rank_audc, rank_aurc)[0]
# spearman_auic_r = spearmanr(rank_auic_r, rank_aurc)[0]
# spearman_auic_masked = spearmanr(rank_auic_masked, rank_aurc)[0]
print("\nSpearman Rank Correlation:")
print(f"LRC vs AURC: {spearman_lrc:.4f}")
print(f"AUDC vs AURC: {spearman_audc:.4f}")
# print(f"AUIC Masked vs AUIC Repainted: {spearman_auic_masked:.4f}")

